"""GPU0-only diagnostic layers; synthetic lower layers are not mining kernels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,ctypes as C,json,subprocess,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','bench']);args=p.parse_args()
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
if uuid in apps:raise SystemExit('GPU0 has another compute process')
import cuda_capi as cc
P,I=C.c_void_p,C.c_int
lib=C.CDLL(str(root/'artifacts/libpearl-rank128-stage-rotation.so'));lib.stage_rotation.argtypes=[I,P,P,I,I,P,P]
ref=C.CDLL(str(root/'artifacts/libpearl-rank.so'));ref.rank_fold.argtypes=[I,P,P,I,I,P,P]
resources={}
for v in range(4):
 info=np.zeros(5,np.int32);cc._chk(lib.stage_rotation(v,None,None,8192,8192,None,info.ctypes.data),'info');resources[v]=info.tolist()
assert all(x[3]==1 and x[4]==0 for x in resources.values()),resources
print('resources',resources,flush=True)
def launch(v,a,b,m,n,t):cc._chk(lib.stage_rotation(v,a.ptr,b.ptr,m,n,t.ptr,None),'launch')
def cpu_sink(acc,m,n):
 gx=m//256;gy=(n+191)//192
 w=np.arange(12)[:,None,None,None,None];r=np.arange(4)[None,:,None,None,None]
 g=np.arange(4)[None,None,:,None,None];l=np.arange(32)[None,None,None,:,None];e=np.arange(8)[None,None,None,None,:]
 ri=(w%4)*64+r*16+l//4+(e%4//2)*8
 ci=(w//4)*64+g*16+(e//4)*8+2*(l%4)+(e%2)
 outputs=[]
 for linear in range(gx*gy):
  first=linear//(8*gx)*8;cols=min(8,gy-first);offset=linear%(8*gx);cm=offset//cols;cn=first+offset%cols
  values=acc[cm*256+ri,cn*192+ci].view(np.uint32)
  outputs.append(np.add.reduce(np.add.reduce(np.add.reduce(values,axis=-1,dtype=np.uint32),axis=2,dtype=np.uint32),axis=1,dtype=np.uint32).ravel())
 return np.concatenate(outputs)
if args.mode=='validate':
 results=[]
 for m,n in ((256,384),(512,400)):
  for seed in (42,1234):
   rng=np.random.default_rng(seed);ah=rng.integers(-128,128,(m,4096),np.int8);bh=rng.integers(-128,128,(n,4096),np.int8)
   padded=np.zeros(((n+191)//192*192,4096),np.int8);padded[:n]=bh
   matrices=[(ah[:,:64].astype(np.int32)@padded[:,:64].astype(np.int32).T)*64,
             (ah[:,:128].astype(np.int32)@padded[:,:128].astype(np.int32).T)*32,
             ah.astype(np.int32)@padded.astype(np.int32).T]
   a,b,t=cc.DBuf(ah.nbytes),cc.DBuf(bh.nbytes),cc.DBuf(max(m*n//256*64,(m//256)*((n+191)//192)*384*4))
   a.from_host(ah);b.from_host(bh)
   cc._chk(ref.rank_fold(128,a.ptr,b.ptr,m,n,t.ptr,None),'reference');et=np.empty(m*n//256*16,np.uint32);t.to_host(et)
   for v in range(4):
    expected=et if v==3 else cpu_sink(matrices[v],m,n)
    t.memset(0xa5);launch(v,a,b,m,n,t);actual=np.empty_like(expected);t.to_host(actual)
    assert np.array_equal(actual,expected),(v,m,n,seed,np.flatnonzero(actual!=expected)[:8])
    assert np.count_nonzero(expected)>0,(v,'degenerate checksum')
    row=dict(variant=v,m=m,n=n,seed=seed,pass_bit_exact=True,nonzero_words=int(np.count_nonzero(actual)));results.append(row);print(row,flush=True)
   for buf in (a,b,t):buf.free()
 (root/'artifacts/rank128-stage-rotation-correctness.json').write_text(json.dumps(results,indent=2)+'\n')
else:
 runtime=C.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so')
 runtime.cudaEventCreate.argtypes=[C.POINTER(P)];runtime.cudaEventRecord.argtypes=[P,P];runtime.cudaEventSynchronize.argtypes=[P]
 runtime.cudaEventElapsedTime.argtypes=[C.POINTER(C.c_float),P,P];runtime.cudaEventDestroy.argtypes=[P]
 m=n=8192;a,b,t=cc.DBuf(m*4096),cc.DBuf(n*4096),cc.DBuf(m*n//256*64)
 rng=np.random.default_rng(42)
 for buf in (a,b):buf.from_host(rng.integers(-128,128,(m,4096),np.int8))
 cc._chk(ref.rank_fold(128,a.ptr,b.ptr,m,n,t.ptr,None),'reference');expected=np.empty(m*n//256*16,np.uint32);t.to_host(expected)
 launch(3,a,b,m,n,t);actual=np.empty_like(expected);t.to_host(actual);assert np.array_equal(actual,expected)
 start,end=P(),P();cc._chk(runtime.cudaEventCreate(C.byref(start)),'event');cc._chk(runtime.cudaEventCreate(C.byref(end)),'event')
 warm=time.perf_counter()
 while time.perf_counter()-warm<3:
  for _ in range(16):launch(3,a,b,m,n,t)
  cc.sync()
 rows=[]
 for repeat in range(8):
  for j in range(4):
   v=(repeat+j)%4
   for _ in range(100):launch(v,a,b,m,n,t)
   cc.sync();values=[]
   for _ in range(25):
    cc._chk(runtime.cudaEventRecord(start,None),'event')
    for _ in range(16):launch(v,a,b,m,n,t)
    cc._chk(runtime.cudaEventRecord(end,None),'event');cc._chk(runtime.cudaEventSynchronize(end),'event');ms=C.c_float();cc._chk(runtime.cudaEventElapsedTime(C.byref(ms),start,end),'event')
    values.append(ms.value/16)
   row=dict(variant=v,repeat=repeat,samples_ms=values,mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
   row['telemetry_end']=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip()
   rows.append(row);print(json.dumps({k:x for k,x in row.items() if k!='samples_ms'}),flush=True)
 result=dict(resources=resources,measurements=rows,summaries={})
 for v in range(4):
  vals=np.array([x for r in rows if r['variant']==v for x in r['samples_ms']]);mean=float(vals.mean())
  result['summaries'][v]=dict(samples=len(vals),mean_ms=mean,median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),useful_work_equiv_TMAC_s=(m*n*4096)/mean/1e9)
 (root/'artifacts/rank128-stage-rotation-events.json').write_text(json.dumps(result,indent=2)+'\n')
 for x in (start,end):cc._chk(runtime.cudaEventDestroy(x),'event destroy')
 for buf in (a,b,t):buf.free()
 print(json.dumps(result['summaries'],indent=2),flush=True)
