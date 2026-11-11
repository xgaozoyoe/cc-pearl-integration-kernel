"""GPU0-only full-A-tile fast path R128 test. Production library is never replaced."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,ctypes as C,json,re,statistics,sys,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
import cuda_capi as cc
import miner_capi as miner
import pool_common
P,I=C.c_void_p,C.c_int
lib=C.CDLL(str(root/'artifacts/libpearl-rank128-ldsm-invariant-experimental.so'))
lib.ldsm_invariant_fold.argtypes=[I,P,P,I,I,P,P]
lib.ldsm_invariant_pow.argtypes=[I,P,P,I,I,P,P,P,P,P,P]
ref=C.CDLL(str(root/'artifacts/libpearl-rank.so'))
ref.rank_pow.argtypes=[I,P,P,I,I,P,P,P,P,P,P]
def pow_call(v,a,b,m,n,key,target,t,d=None,f=None,c=None):
 cc._chk(lib.ldsm_invariant_pow(v,cc._as(a),cc._as(b),m,n,cc._as(key),cc._as(target),cc._as(t),cc._as(d),cc._as(f),cc._as(c)),'xor control pow')
p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','bench','profile']);p.add_argument('--variant',type=int,default=0);p.add_argument('--small-only',action='store_true');args=p.parse_args()
resources={}
for v in (0,1,2):
 info=np.zeros(5,np.int32);cc._chk(lib.ldsm_invariant_fold(v,None,None,8192,8192,None,info.ctypes.data),'info');resources[v]=info.tolist()
print('resources',resources,flush=True)
if args.mode=='validate':
 results=[]
 for m,n in (((256,48),(256,384),(256,400)) if args.small_only else ((256,48),(256,384),(256,400),(8192,8192))):
  for seed in (42,1234):
   rng=np.random.default_rng(seed);alloc=[]
   def buf(size):
    x=cc.DBuf(size);alloc.append(x);return x
   a,b=buf(m*4096),buf(n*4096);a.from_host(rng.integers(-128,128,(m,4096),np.int8));b.from_host(rng.integers(-128,128,(n,4096),np.int8))
   key,target=buf(32),buf(32);key.from_host(rng.integers(0,256,32,np.uint8));target.memset(0)
   t,d=buf(m*n//256*64),buf(m*n//256*32)
   cc._chk(ref.rank_pow(128,a.ptr,b.ptr,m,n,key.ptr,target.ptr,t.ptr,d.ptr,None,None),'frozen')
   expected_t=np.empty(m*n//256*16,np.uint32);expected_d=np.empty(m*n//256*32,np.uint8)
   t.to_host(expected_t);d.to_host(expected_d)
   for v in (0,1,2):
    t.memset(0xa5);d.memset(0xa5);pow_call(v,a,b,m,n,key,target,t,d)
    actual_t=np.empty_like(expected_t);actual_d=np.empty_like(expected_d);t.to_host(actual_t);d.to_host(actual_d)
    assert np.array_equal(actual_t,expected_t),(v,m,n,seed,'transcript')
    assert np.array_equal(actual_d,expected_d),(v,m,n,seed,'digest')
    row=dict(variant=v,m=m,n=n,seed=seed,transcript_pass=True,digest_pass=True);results.append(row);print('PASS',row,flush=True)
   for x in alloc:x.free()
 Path('artifacts/rank128-ldsm-invariant-correctness-small.json' if args.small_only else 'artifacts/rank128-ldsm-invariant-correctness.json').write_text(json.dumps(results,indent=2)+'\n')
elif args.mode=='profile':
 a,b,t=cc.DBuf(8192*4096),cc.DBuf(8192*4096),cc.DBuf(512*512*64)
 rng=np.random.default_rng(42)
 for x in (a,b):x.from_host(rng.integers(-128,128,(8192,4096),np.int8))
 for _ in range(40):
  cc._chk(lib.ldsm_invariant_fold(args.variant,a.ptr,b.ptr,8192,8192,t.ptr,None),'profile');cc.sync()
else:
 assert all(x[3]==1 and x[4]==0 for x in resources.values()),resources
 miner.PROOF_PROCESS=False;miner.R=pool_common.R=128
 class Pool:
  def check_newer_job(self,job):return None
 class Scheduler:
  mode='user'
  def __init__(self):self.grids=0
  def note(self,elapsed):pass
  def maybe_switch(self):self.grids+=1;return self.grids>=12
 original_setup=cc.setup_job;results=[]
 for repeat in range(6):
  for v in tuple((repeat+j)%3 for j in range(3)):
   scheduler=Scheduler();times=[];bufs=miner.Bufs(8192)
   cc.setup_job=lambda *a:original_setup(*a[:-1],202610070000+scheduler.grids)
   cc.pearl_pow_split=lambda a,b,m,n,k,r,key,target,t,dig,found,coord,variant:pow_call(v,a,b,m,n,key,target,t,dig,found,coord)
   def log(line):
    match=re.search(r'grid wall: ([\d.]+) s',line)
    if match:times.append(float(match[1])*1000)
   miner.mine_job(Pool(),pool_common.real_config(),bytes(80),0,'offline',8192,0,scheduler,bufs,dict(user=0,dev=0),None,log)
   values=times[2:];mean=statistics.mean(values)
   row=dict(variant=v,repeat=repeat,samples=len(values),median_ms=statistics.median(values),mean_ms=mean,p95_ms=float(np.percentile(values,95)),grids_sec=1000/mean,consensus_equivalent_ths=(2**46/1e12)*1000/mean)
   row["telemetry_end"]=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip()
   row["samples_ms"]=values;results.append(row);print(json.dumps({k:x for k,x in row.items() if k!="samples_ms"}),flush=True)
   for value in vars(bufs).values():
    if isinstance(value,cc.DBuf):value.free()
    elif isinstance(value,list):
     for x in value:
      if isinstance(x,cc.DBuf):x.free()
 Path('artifacts/rank128-ldsm-invariant-results.json').write_text(json.dumps(dict(resources=resources,measurements=results),indent=2)+'\n')
