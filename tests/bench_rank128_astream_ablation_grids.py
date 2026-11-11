"""GPU0 real offline full-grid A-stream ablation; PoW disabled in all variants."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import ctypes as C,json,re,statistics,sys,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
if uuid in apps:raise SystemExit('GPU0 busy')
import cuda_capi as cc
import miner_capi as miner
import pool_common
lib=C.CDLL(str(root/'artifacts/libpearl-rank128-astream-ablation.so'))
lib.astream_ablation.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p]
def fold(v,a,b,m,n,t):
 cc._chk(lib.astream_ablation(v,cc._as(a),cc._as(b),m,n,cc._as(t),None),'ablation')
resources={}
for v in (0,1,4):
 info=np.zeros(5,np.int32);cc._chk(lib.astream_ablation(v,None,None,8192,8192,None,info.ctypes.data),'info');resources[v]=info.tolist()
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
 for v in tuple((0,1,4)[(repeat+j)%3] for j in range(3)):
  scheduler=Scheduler();times=[];bufs=miner.Bufs(8192)
  cc.setup_job=lambda *a:original_setup(*a[:-1],202610070000+scheduler.grids)
  cc.pearl_pow_split=lambda a,b,m,n,k,r,key,target,t,dig,found,coord,variant:fold(v,a,b,m,n,t)
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
Path('artifacts/rank128-astream-ablation-grids.json').write_text(json.dumps(dict(resources=resources,measurements=results),indent=2)+'\n')
