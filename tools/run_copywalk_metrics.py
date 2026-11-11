"""Offline full-grid load for external Tensor Active metrics. GPU0 only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,ctypes as C,time,sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(root/'python'))
import cuda_capi as cc
import miner_capi as miner
import pool_common
lib=C.CDLL(str(args.library.resolve()));lib.rank_pow.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int]+[C.c_void_p]*6
miner.PROOF_PROCESS=False;miner.R=pool_common.R=128
class Pool:
 def check_newer_job(self,job):return None
class Scheduler:
 mode='user'
 def __init__(self):self.grids=0;self.start=time.perf_counter()
 def note(self,elapsed):pass
 def maybe_switch(self):self.grids+=1;return time.perf_counter()-self.start>=42
scheduler=Scheduler();setup=cc.setup_job
cc.setup_job=lambda *a:setup(*a[:-1],202610070000+scheduler.grids)
def pow_call(a,b,m,n,k,r,key,target,t,d,found,coord,variant):
 assert k==4096 and r==128
 cc._chk(lib.rank_pow(r,cc._as(a),cc._as(b),m,n,cc._as(key),cc._as(target),cc._as(t),cc._as(d),cc._as(found),cc._as(coord)),'rank_pow')
cc.pearl_pow_split=pow_call
bufs=miner.Bufs(8192)
miner.mine_job(Pool(),pool_common.real_config(),bytes(80),0,'offline-metrics',8192,0,scheduler,bufs,dict(user=0,dev=0),None,lambda x:print(x,flush=True))
print(json.dumps(dict(grids=scheduler.grids,seconds=time.perf_counter()-scheduler.start,library=str(args.library))))
