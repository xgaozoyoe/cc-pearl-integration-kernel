"""Full per-K preparation replay controls; GPU0 only, never a miner backend."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import ctypes as C,argparse,json,sys,time,subprocess
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'python'),str(root/'tools')]
from check_gpu0_idle import check_gpu0_idle
p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','bench','profile']);p.add_argument('--variant',type=int,choices=range(5),default=2);p.add_argument('--small-only',action='store_true');p.add_argument('--allow-supervisor-pid',type=int,default=0);args=p.parse_args()
if args.mode!='profile':check_gpu0_idle(args.allow_supervisor_pid)
import cuda_capi as cc
P,I=C.c_void_p,C.c_int
lib=C.CDLL(str(root/'artifacts/libpearl-abcd-scope.so'));lib.scope_fold.argtypes=[I,P,P,I,I,P,P,P,P]
ref=C.CDLL(str(root/'artifacts/libpearl-rank.so'));ref.rank_fold.argtypes=[I,P,P,I,I,P,P]
names=['A','B','B_matched','C_replay','D_reduced'];resources={}
for v,n in enumerate(names):
 info=np.zeros(5,np.int32);cc._chk(lib.scope_fold(v,None,None,8192,8192,None,info.ctypes.data,None,None),'resources');resources[n]=info.tolist()
print('resources [regs,threads,shared,CTAs/SM,local]',resources,flush=True)
def launch(v,a,b,m,n,t,trace,audit):cc._chk(lib.scope_fold(v,a.ptr,b.ptr,m,n,t.ptr,None,trace.ptr,audit.ptr),'scope fold')
def setup(m,n,seed):
 rng=np.random.default_rng(seed);a,b,t=cc.DBuf(m*4096),cc.DBuf(n*4096),cc.DBuf(m*n//256*64)
 for buf,rows in ((a,m),(b,n)):buf.from_host(rng.integers(-128,128,(rows,4096),np.int8))
 size=((m+255)//256)*((n+191)//192)*384*65*41*4
 trace,audit=cc.DBuf(size),cc.DBuf(size);trace.memset(0);audit.memset(0)
 cc._chk(ref.rank_fold(128,a.ptr,b.ptr,m,n,t.ptr,None),'reference');expected=np.empty(m*n//256*16,np.uint32);t.to_host(expected)
 launch(5,a,b,m,n,t,trace,trace);got=np.empty_like(expected);t.to_host(got);assert np.array_equal(got,expected),'record transcript'
 return a,b,t,trace,audit,expected
if args.mode=='validate':
 results=[]
 shapes=[(256,48),(256,384),(256,400)] + ([] if args.small_only else [(8192,8192)])
 for m,n in shapes:
  for seed in (42,1234):
   a,b,t,trace,audit,expected=setup(m,n,seed)
   if m==256:
    plan=np.empty(trace.nbytes//4,np.uint32) if hasattr(trace,'nbytes') else np.empty(((m+255)//256)*((n+191)//192)*384*65*41,np.uint32)
    trace.to_host(plan)
   for v,name in enumerate(names):
    t.memset(0xa5);audit.memset(0);launch(v,a,b,m,n,t,trace,audit);got=np.empty_like(expected);t.to_host(got);assert np.array_equal(got,expected),(name,m,n,seed,'transcript')
    if m==256 and v>=2:
     actual=np.empty_like(plan);audit.to_host(actual);assert np.array_equal(actual,plan),(name,m,n,seed,'prepared values',np.flatnonzero(actual!=plan)[:5])
    row=dict(variant=name,m=m,n=n,seed=seed,transcript_pass=True,prepared_values_pass=(m==256 and v>=2));results.append(row);print(row,flush=True)
   for buf in (a,b,t,trace,audit):buf.free()
 (root/'artifacts/abcd-scope-validation.json').write_text(json.dumps(results,indent=2)+'\n')
else:
 m=n=8192;a,b,t,trace,audit,expected=setup(m,n,42)
 if args.mode=='profile':
  for _ in range(36):launch(args.variant,a,b,m,n,t,trace,audit);cc.sync()
  sys.exit(0)
 for v in range(5):
  launch(v,a,b,m,n,t,trace,audit);got=np.empty_like(expected);t.to_host(got);assert np.array_equal(got,expected),names[v]
 rt=C.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so');rt.cudaEventCreate.argtypes=[C.POINTER(P)];rt.cudaEventRecord.argtypes=[P,P];rt.cudaEventSynchronize.argtypes=[P];rt.cudaEventElapsedTime.argtypes=[C.POINTER(C.c_float),P,P]
 start,end=P(),P();cc._chk(rt.cudaEventCreate(C.byref(start)),'event');cc._chk(rt.cudaEventCreate(C.byref(end)),'event')
 until=time.perf_counter()+3
 while time.perf_counter()<until:
  for v in range(5):launch(v,a,b,m,n,t,trace,audit)
  cc.sync()
 rows=[]
 for repeat in range(4):
  for j in range(5):
   v=(j+repeat)%5
   for _ in range(10):launch(v,a,b,m,n,t,trace,audit)
   cc.sync();values=[]
   for _ in range(10):
    cc._chk(rt.cudaEventRecord(start,None),'start');launch(v,a,b,m,n,t,trace,audit);cc._chk(rt.cudaEventRecord(end,None),'end');cc._chk(rt.cudaEventSynchronize(end),'wait');ms=C.c_float();cc._chk(rt.cudaEventElapsedTime(C.byref(ms),start,end),'ms');values.append(ms.value)
   row=dict(name=names[v],repeat=repeat,samples_ms=values,mean_ms=float(np.mean(values)));rows.append(row);print(row,flush=True)
 result=dict(resources=resources,measurements=rows,summaries={})
 for n in names:
  vals=np.array([v for row in rows if row['name']==n for v in row['samples_ms']]);result['summaries'][n]=dict(mean_ms=float(vals.mean()),median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),samples=len(vals))
 result['instrumentation_time_ratio']=result['summaries']['B_matched']['mean_ms']/result['summaries']['B']['mean_ms']
 result['native_attribution_valid']=False # Replay adds global RAW/storage effects; quantify, never silently treat as native.
 (root/'artifacts/abcd-scope-events.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summaries'],indent=2));print('instrumentation time ratio',result['instrumentation_time_ratio'])
