"""GPU0-only correct-arithmetic early-arrival diagnostic, never a mining backend."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import ctypes as C,json,subprocess,sys,argparse,time,hashlib
import numpy as np
root=Path(__file__).resolve().parents[1];out=root/'artifacts/fold-defer-reduce';sys.path.insert(0,str(root/'python'))
p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','bench','profile']);p.add_argument('--variant',type=int,default=0,choices=range(5));p.add_argument('--allow-supervisor-pid',type=int,default=0);p.add_argument('--names',nargs='+');args=p.parse_args()
sys.path.insert(0,str(root/'tools'))
from check_gpu0_idle import check_gpu0_idle
if args.mode!='profile':check_gpu0_idle(args.allow_supervisor_pid)
import cuda_capi as cc
P,I,U=C.c_void_p,C.c_int,C.c_uint
# Force runtime primary-context creation before using the driver API.
init=cc.DBuf(4)
drv=C.CDLL('libcuda.so.1');drv.cuInit.argtypes=[U];drv.cuModuleLoad.argtypes=[C.POINTER(P),C.c_char_p];drv.cuModuleGetFunction.argtypes=[C.POINTER(P),P,C.c_char_p]
drv.cuFuncSetAttribute.argtypes=[P,I,I];drv.cuFuncGetAttribute.argtypes=[C.POINTER(I),I,P];drv.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes=[C.POINTER(I),P,I,C.c_size_t]
drv.cuLaunchKernel.argtypes=[P,U,U,U,U,U,U,U,P,C.POINTER(P),P]
def check(x):
 if x:raise RuntimeError('CUDA driver status '+str(x))
check(drv.cuInit(0));manifest=json.loads((out/'manifest.json').read_text());names=args.names or list(manifest['variants']);assert len(set(names))==len(names) and all(n in manifest['variants'] for n in names);mods=[];funcs=[];resources={}
for name in names:
 assert hashlib.sha256((out/(name+'.cubin')).read_bytes()).hexdigest()==manifest['variants'][name]['sha256']
 path=out/(name+'.cubin')
 mod,f=P(),P();check(drv.cuModuleLoad(C.byref(mod),str(path).encode()));check(drv.cuModuleGetFunction(C.byref(f),mod,manifest['variants'][name]['entry'].encode()))
 check(drv.cuFuncSetAttribute(f,8,manifest['variants'][name]['smem']));check(drv.cuFuncSetAttribute(f,9,100))
 info=[]
 for key in (4,3,1):
  val=I();check(drv.cuFuncGetAttribute(C.byref(val),key,f));info.append(val.value)
 active=I();check(drv.cuOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(active),f,384,manifest['variants'][name]['smem']));info.append(active.value)
 assert info[0]<=168 and info[2:]==[0,1],info
 resources[name]=info;mods.append(mod);funcs.append(f)
print('resources [registers,local,static_shared,CTAs/SM]',resources,flush=True)
def launch(v,a,b,m,n,t):
 vals=[P(a.ptr.value if isinstance(a.ptr,P) else a.ptr),P(b.ptr.value if isinstance(b.ptr,P) else b.ptr),I(m),I(n),I(4096),I(128),P(t.ptr.value if isinstance(t.ptr,P) else t.ptr)]
 pars=(P*len(vals))(*[C.cast(C.byref(x),P) for x in vals]);check(drv.cuLaunchKernel(funcs[v],(m+255)//256,(n+191)//192,1,384,1,1,manifest['variants'][names[v]]['smem'],None,pars,None))
ref=C.CDLL(str(root/'artifacts/libpearl-rank.so'));ref.rank_fold.argtypes=[I,P,P,I,I,P,P]
if args.mode=='validate':
 rows=[]
 for m,n in ((256,384),(256,400),(512,768),(512,800)):
  for seed in (42,1234,9876):
   rng=np.random.default_rng(seed);ah=rng.integers(-128,128,(m,4096),np.int8);bh=rng.integers(-128,128,(n,4096),np.int8)
   a,b,t=cc.DBuf(ah.nbytes),cc.DBuf(bh.nbytes),cc.DBuf(m*n//256*64);a.from_host(ah);b.from_host(bh)
   cc._chk(ref.rank_fold(128,a.ptr,b.ptr,m,n,t.ptr,None),'reference');expected=np.empty(m*n//256*16,np.uint32);t.to_host(expected)
   for v,name in enumerate(names):
    t.memset(0xa5);launch(v,a,b,m,n,t);got=np.empty_like(expected);t.to_host(got)
    exp=expected if manifest['variants'][name]['correct'] else np.zeros_like(expected)
    assert np.array_equal(got,exp),(name,m,n,seed,np.flatnonzero(got!=exp)[:8])
    row=dict(name=name,m=m,n=n,seed=seed,expected_probe_semantics_pass=True,transcript_bit_exact=manifest['variants'][name]['correct']);rows.append(row);print(row,flush=True)
   for buf in (a,b,t):buf.free()
 (out/'validation.json').write_text(json.dumps(rows,indent=2)+'\n')
else:
 m=n=8192;a,b,t=cc.DBuf(m*4096),cc.DBuf(n*4096),cc.DBuf(m*n//256*64);rng=np.random.default_rng(42)
 for buf in (a,b):buf.from_host(rng.integers(-128,128,(m,4096),np.int8))
 if args.mode=='profile':
  for _ in range(40):launch(args.variant,a,b,m,n,t);cc.sync()
  sys.exit(0)
 cc._chk(ref.rank_fold(128,a.ptr,b.ptr,m,n,t.ptr,None),'reference');expected=np.empty(m*n//256*16,np.uint32);t.to_host(expected)
 for v in range(len(names)):
  launch(v,a,b,m,n,t);got=np.empty_like(expected);t.to_host(got);assert np.array_equal(got,expected if manifest['variants'][names[v]]['correct'] else np.zeros_like(expected)),names[v]
 runtime=C.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so');runtime.cudaEventCreate.argtypes=[C.POINTER(P)];runtime.cudaEventRecord.argtypes=[P,P];runtime.cudaEventSynchronize.argtypes=[P];runtime.cudaEventElapsedTime.argtypes=[C.POINTER(C.c_float),P,P];runtime.cudaEventDestroy.argtypes=[P]
 start,end=P(),P();cc._chk(runtime.cudaEventCreate(C.byref(start)),'event');cc._chk(runtime.cudaEventCreate(C.byref(end)),'event')
 warm=time.perf_counter()
 while time.perf_counter()-warm<3:
  for _ in range(16):launch(0,a,b,m,n,t)
  cc.sync()
 rows=[]
 for repeat in range(10):
  for j in range(len(names)):
   v=(repeat+j)%len(names)
   for _ in range(100):launch(v,a,b,m,n,t)
   cc.sync();values=[]
   for _ in range(20):
    cc._chk(runtime.cudaEventRecord(start,None),'event')
    for _ in range(16):launch(v,a,b,m,n,t)
    cc._chk(runtime.cudaEventRecord(end,None),'event');cc._chk(runtime.cudaEventSynchronize(end),'event');ms=C.c_float();cc._chk(runtime.cudaEventElapsedTime(C.byref(ms),start,end),'event');values.append(ms.value/16)
   row=dict(name=names[v],variant=v,repeat=repeat,samples_ms=values,mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
   row['telemetry_end']=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip();rows.append(row);print(json.dumps({k:x for k,x in row.items() if k!='samples_ms'}),flush=True)
 result=dict(resources=resources,measurements=rows,summaries={})
 for name in names:
  vals=np.array([x for r in rows if r['name']==name for x in r['samples_ms']]);result['summaries'][name]=dict(samples=len(vals),mean_ms=float(vals.mean()),median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)))
 (out/'events.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summaries'],indent=2),flush=True)
 for x in (start,end):cc._chk(runtime.cudaEventDestroy(x),'event destroy')
 for buf in (a,b,t):buf.free()
