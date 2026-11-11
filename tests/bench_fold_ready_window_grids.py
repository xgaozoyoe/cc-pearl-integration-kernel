"""GPU0-only correct-arithmetic early-arrival diagnostic, never a mining backend."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import ctypes as C,json,subprocess,sys,argparse,time,hashlib
import numpy as np
root=Path(__file__).resolve().parents[1];out=root/'artifacts/fold-ready-window';sys.path.insert(0,str(root/'python'))
p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','grids']);p.add_argument('--variant',type=int,default=0,choices=range(3));p.add_argument('--allow-supervisor-pid',type=int,default=0);p.add_argument('--names',nargs='+');p.add_argument('--repeats',type=int,default=6);args=p.parse_args();assert args.repeats>0
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

import statistics,re
lib=C.CDLL(str(root/'artifacts/libpearl-handoff-lean.so'))
blake=getattr(lib,'_Z19launch_pearl_blake3PKjiiS0_S0_PhPiS2_P11CUstream_st')
blake.argtypes=[P,I,I,P,P,P,P,P,P];blake.restype=None
ref.rank_pow.argtypes=[I,P,P,I,I,P,P,P,P,P,P]
def pow_call(v,a,b,m,n,key,target,t,d=None,f=None,c=None):
 launch(v,a,b,m,n,t)
 blake(cc._as(t),(m//16)*(n//16),n,cc._as(key),cc._as(target),cc._as(d),cc._as(f),cc._as(c),None)
if args.mode=='validate':
 results=[]
 for m,n in ((256,48),(256,400),(8192,8192)):
  for seed in (42,1234):
   rng=np.random.default_rng(seed);a,b,t,d=cc.DBuf(m*4096),cc.DBuf(n*4096),cc.DBuf(m*n//256*64),cc.DBuf(m*n//256*32)
   key,target=cc.DBuf(32),cc.DBuf(32);key.from_host(rng.integers(0,256,32,np.uint8));target.memset(0)
   a.from_host(rng.integers(-128,128,(m,4096),np.int8));b.from_host(rng.integers(-128,128,(n,4096),np.int8))
   cc._chk(ref.rank_pow(128,a.ptr,b.ptr,m,n,key.ptr,target.ptr,t.ptr,d.ptr,None,None),'reference pow')
   et=np.empty(m*n//256*16,np.uint32);ed=np.empty(m*n//256*32,np.uint8);t.to_host(et);d.to_host(ed)
   for v,name in enumerate(names):
    t.memset(0xa5);d.memset(0xa5);pow_call(v,a,b,m,n,key,target,t,d)
    at=np.empty_like(et);ad=np.empty_like(ed);t.to_host(at);d.to_host(ad);assert np.array_equal(at,et),(name,'transcript');assert np.array_equal(ad,ed),(name,'digest')
    row=dict(name=name,m=m,n=n,seed=seed,transcript_pass=True,digest_pass=True);results.append(row);print(row,flush=True)
   for buf in (a,b,t,d,key,target):buf.free()
 (out/'pow-validation.json').write_text(json.dumps(results,indent=2)+'\n')
else:
 import miner_capi as miner
 import pool_common
 miner.PROOF_PROCESS=False;miner.R=pool_common.R=128
 class Pool:
  def check_newer_job(self,job):return None
 class Scheduler:
  mode='user'
  def __init__(self):self.grids=0
  def note(self,elapsed):pass
  def maybe_switch(self):self.grids+=1;return self.grids>=12
 original_setup=cc.setup_job;results=[]
 for repeat in range(args.repeats):
  for j in range(len(names)):
   v=(repeat+j)%len(names);scheduler=Scheduler();times=[];bufs=miner.Bufs(8192)
   cc.setup_job=lambda *a:original_setup(*a[:-1],202610100000+scheduler.grids)
   cc.pearl_pow_split=lambda a,b,m,n,k,r,key,target,t,dig,found,coord,variant:pow_call(v,a,b,m,n,key,target,t,dig,found,coord)
   def log(line):
    match=re.search(r'grid wall: ([\d.]+) s',line)
    if match:times.append(float(match[1])*1000)
   miner.mine_job(Pool(),pool_common.real_config(),bytes(80),0,'offline',8192,0,scheduler,bufs,dict(user=0,dev=0),None,log)
   values=times[2:];mean=statistics.mean(values)
   row=dict(name=names[v],variant=v,repeat=repeat,samples_ms=values,mean_ms=mean,median_ms=statistics.median(values),p95_ms=float(np.percentile(values,95)),grids_sec=1000/mean,consensus_equivalent_ths=(2**46/1e12)*1000/mean)
   row['telemetry_end']=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip()
   results.append(row);print(json.dumps({k:x for k,x in row.items() if k!='samples_ms'}),flush=True)
   for value in vars(bufs).values():
    if isinstance(value,cc.DBuf):value.free()
    elif isinstance(value,list):
     for x in value:
      if isinstance(x,cc.DBuf):x.free()
 (out/'grids.json').write_text(json.dumps(dict(resources=resources,measurements=results),indent=2)+'\n')
