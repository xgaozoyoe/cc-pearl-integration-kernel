"""GPU0 offline R128 fold ablation, same inputs/mapping/staging in both modes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,ctypes as C,json,re,statistics,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
import cuda_capi as cc
import miner_capi as miner
import pool_common
p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','bench','profile']);p.add_argument('--variant',type=int,default=0);args=p.parse_args()
P,I=C.c_void_p,C.c_int
lib=C.CDLL('/tmp/libpearl-rank128-ablation.so')
lib.ablation.argtypes=[I,P,P,I,I,P,P]
lib.event_time.argtypes=[I,P,P,I,I,P,I,P]
def launch(v,a,b,m,n,out):cc._chk(lib.ablation(v,a.ptr,b.ptr,m,n,out.ptr,None),'ablation')
resources={}
for v in (0,1):
 info=np.zeros(5,np.int32);cc._chk(lib.ablation(v,None,None,8192,8192,None,info.ctypes.data),'info');resources[v]=info.tolist()
print('resources',resources,flush=True)

if args.mode=='validate':
 ref=C.CDLL(str(root/'artifacts/libpearl-rank.so'));ref.rank_fold.argtypes=[I,P,P,I,I,P,P]
 for seed in (42,1234):
  rng=np.random.default_rng(seed);m,n=256,384
  a=rng.integers(-128,128,(m,4096),np.int8);b=rng.integers(-128,128,(n,4096),np.int8)
  da,db,out=cc.DBuf(a.nbytes),cc.DBuf(b.nbytes),cc.DBuf(m*n//256*64)
  da.from_host(a);db.from_host(b)
  expected=np.empty(m*n//256*16,np.uint32);actual=np.empty_like(expected)
  cc._chk(ref.rank_fold(128,da.ptr,db.ptr,m,n,out.ptr,None),'frozen');out.to_host(expected)
  out.memset(0xa5);launch(0,da,db,m,n,out);out.to_host(actual)
  assert np.array_equal(expected,actual),'full transcript changed'
  out.memset(0xa5);launch(1,da,db,m,n,out);out.to_host(actual)
  sink=actual[:((m+255)//256)*((n+191)//192)*384]
  expected_xor=np.bitwise_xor.reduce((a.astype(np.int32)@b.astype(np.int32).T).view(np.uint32).ravel())
  assert np.bitwise_xor.reduce(sink)==expected_xor,'pure accumulator checksum'
  print('PASS frozen transcript + CPU final-accumulator checksum',seed,flush=True)
  for x in (da,db,out):x.free()
else:
 m=n=8192;da,db,out=cc.DBuf(m*4096),cc.DBuf(n*4096),cc.DBuf(m*n//256*64)
 rng=np.random.default_rng(42)
 da.from_host(rng.integers(-128,128,(m,4096),np.int8));db.from_host(rng.integers(-128,128,(n,4096),np.int8))
 if args.mode=='profile':
  for _ in range(40):launch(args.variant,da,db,m,n,out);cc.sync()
  sys.exit(0)
 results=[]
 for repeat in range(2):
  for v in ((0,1) if repeat==0 else (1,0)):
   for _ in range(40):launch(v,da,db,m,n,out)
   cc.sync();values=[]
   for _ in range(40):
    ms=C.c_float();cc._chk(lib.event_time(v,da.ptr,db.ptr,m,n,out.ptr,16,C.byref(ms)),'events');values.append(ms.value/16)
   row=dict(type='kernel_events',variant=v,repeat=repeat,median_ms=statistics.median(values),mean_ms=statistics.mean(values),p95_ms=float(np.percentile(values,95)),region=8192)
   results.append(row);print(json.dumps(row),flush=True)
 for x in (da,db,out):x.free()
 miner.PROOF_PROCESS=False;miner.R=pool_common.R=128
 class Pool:
  def check_newer_job(self,job):return None
 class Scheduler:
  mode='user'
  def __init__(self):self.grids=0
  def note(self,elapsed):pass
  def maybe_switch(self):self.grids+=1;return self.grids>=44
 original_setup=cc.setup_job
 for repeat in range(2):
  for v in ((0,1) if repeat==0 else (1,0)):
   scheduler=Scheduler();times=[];bufs=miner.Bufs(8192)
   cc.setup_job=lambda *a:original_setup(*a[:-1],202610070000+scheduler.grids)
   # PoW/jackpot disabled in BOTH controls; same setup/noise/region loop.
   cc.pearl_pow_split=lambda a,b,m,n,k,r,key,target,t,dig,found,coord,variant:launch(v,a,b,m,n,t)
   def log(line):
    match=re.search(r'grid wall: ([\d.]+) s',line)
    if match:times.append(float(match[1])*1000)
   miner.mine_job(Pool(),pool_common.real_config(),bytes(80),0,'offline',8192,0,scheduler,bufs,dict(user=0,dev=0),None,log)
   values=times[4:];mean=statistics.mean(values)
   row=dict(type='grid_no_pow',variant=v,repeat=repeat,median_ms=statistics.median(values),mean_ms=mean,p95_ms=float(np.percentile(values,95)),grids_sec=1000/mean,equivalent_ths=(2**46/1e12)*1000/mean)
   results.append(row);print(json.dumps(row),flush=True)
   for value in vars(bufs).values():
    if isinstance(value,cc.DBuf):value.free()
    elif isinstance(value,list):
     for x in value:
      if isinstance(x,cc.DBuf):x.free()
 Path('artifacts/rank128-ablation-results.json').write_text(json.dumps(dict(resources=resources,measurements=results),indent=2)+'\n')
