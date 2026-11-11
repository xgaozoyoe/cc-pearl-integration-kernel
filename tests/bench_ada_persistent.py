"""Offline GPU0 v121 vs persistent N8 vs fixed-window, region8192."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse,ctypes as C,json,re,sys,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"python"))
import cuda_capi as cc
import miner_capi as miner
from pool_common import real_config
p=argparse.ArgumentParser()
p.add_argument("mode",choices=["validate","bench","profile"])
p.add_argument("--variant",type=int,default=1,choices=range(4))
p.add_argument("--samples",type=int,default=20)
p.add_argument("--runs",type=int,default=2)
p.add_argument("--launches",type=int,default=40)
args=p.parse_args()
P=C.c_void_p;I=C.c_int
lib=C.CDLL("/tmp/libpearl-persistent.so")
lib.persistent_pow.argtypes=[I,P,P,I,I,P,P,P,P,P,P]
lib.persistent_fold.argtypes=[I,P,P,I,I,P,P]
names={0:"v121",3:"fixed8192 ordinary",1:"persistent N8",2:"persistent M16 N8"}
resources={}
for v in names:
 info=np.zeros(5,np.int32)
 cc._chk(lib.persistent_fold(v,None,None,8192,8192,None,info.ctypes.data),"info")
 assert info.tolist()==[168,384,69632,1,0],(v,info)
 resources[v]=info.tolist()
print("resources",resources,flush=True)
def pow(v,a,b,key,target,t,d=None,f=None,c=None):
 cc._chk(lib.persistent_pow(v,cc._as(a),cc._as(b),8192,8192,cc._as(key),cc._as(target),cc._as(t),cc._as(d),cc._as(f),cc._as(c)),"persistent")
if args.mode in ("validate","profile"):
 da,db=cc.DBuf(8192*4096),cc.DBuf(8192*4096)
 key,target=cc.DBuf(32),cc.DBuf(32);target.memset(0)
 t,d=cc.DBuf(512*512*64),cc.DBuf(512*512*32)
 results=[]
 for seed in ([42,1234] if args.mode=="validate" else [42]):
  rng=np.random.default_rng(seed)
  da.from_host(rng.integers(-128,128,(8192,4096),np.int8))
  db.from_host(rng.integers(-128,128,(8192,4096),np.int8))
  key.from_host(rng.integers(0,256,32,np.uint8))
  if args.mode=="profile":
   for _ in range(args.launches):
    cc._chk(lib.persistent_fold(args.variant,da.ptr,db.ptr,8192,8192,t.ptr,None),"profile")
    cc.sync()
   break
  ref=C.CDLL("/tmp/libpearl-ada-linear-copy.so")
  ref.trial_pow.argtypes=[I,P,P,I,I,I,I,P,P,P,P,P,P]
  cc._chk(ref.trial_pow(121,da.ptr,db.ptr,8192,8192,4096,256,key.ptr,target.ptr,t.ptr,d.ptr,None,None),"frozen")
  cc.sync()
  expected_t=np.empty((512,512,16),np.uint32);expected_d=np.empty((512,512,32),np.uint8)
  t.to_host(expected_t);d.to_host(expected_d)
  for v in names:
   t.memset(0xA5);d.memset(0xA5)
   pow(v,da,db,key,target,t,d);cc.sync()
   actual_t=np.empty_like(expected_t);actual_d=np.empty_like(expected_d)
   t.to_host(actual_t);d.to_host(actual_d)
   assert np.array_equal(expected_t,actual_t),(seed,v,"transcript")
   assert np.array_equal(expected_d,actual_d),(seed,v,"digest")
   results.append(dict(seed=seed,variant=v,transcript_pass=True,digest_pass=True))
   print("PASS",results[-1],flush=True)
 if args.mode=="validate":
  (root/"artifacts/ada-persistent-correctness.json").write_text(json.dumps(results,indent=2)+"\n")
else:
 original_pow=cc.pearl_pow_split;original_setup=cc.setup_job
 v=0
 cc.pearl_pow_split=lambda a,b,m,n,k,r,key,target,t,dig,found,coord,variant:pow(v,a,b,key,target,t,dig,found,coord)
 class Pool:
  def check_newer_job(self,job):return None
 class Scheduler:
  mode="user"
  def __init__(self):self.grids=0
  def note(self,elapsed):pass
  def maybe_switch(self):
   self.grids+=1;return self.grids>=args.samples+4
 results=[]
 try:
  for repeat in range(args.runs):
   variants=[0,3,1,2]
   if repeat%2:variants.reverse()
   for v in variants:
    count=[0]
    def setup(*a):
     seed=0x504541524c020000+repeat*(args.samples+4)+count[0];count[0]+=1
     return original_setup(*a[:-1],seed)
    cc.setup_job=setup
    times=[];reported=[];buf=miner.Bufs(8192)
    def log(msg):
     match=re.search(r"grid wall: ([\d.]+) s",msg)
     if match:times.append(float(match[1])*1000)
     match=re.search(r"grid \d+ done:.*\(([\d.]+) TH/s\)",msg)
     if match:reported.append(float(match[1]))
    try:miner.mine_job(Pool(),real_config(),bytes(80),0,"offline-persistent",8192,0,Scheduler(),buf,{"user":0,"dev":0},None,log)
    finally:
     for value in vars(buf).values():
      if isinstance(value,cc.DBuf):value.free()
      elif isinstance(value,list):
       for b in value:b.free()
    telemetry=subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit","--format=csv,noheader"],text=True).strip()
    assert len(times)==args.samples+4
    results.append(dict(run=repeat,variant=v,wall_ms=times,reported_search_THS=reported,telemetry=telemetry))
    print(repeat,names[v],"median",float(np.median(times[4:])),"TH/s",70.368744177664*1000/float(np.median(times[4:])),telemetry,flush=True)
 finally:cc.pearl_pow_split=original_pow;cc.setup_job=original_setup
 summary={}
 for v in names:
  vals=np.array([t for block in results if block["variant"]==v for t in block["wall_ms"][4:]])
  med=float(np.median(vals))
  summary[v]=dict(name=names[v],median_ms=med,mean_ms=float(vals.mean()),p95_ms=float(np.percentile(vals,95)),
   grids_per_sec=1000/med,effective_THS=70.368744177664*1000/med,
   search_THS_median=float(np.median([s for b in results if b["variant"]==v for s in b["reported_search_THS"][4:]])))
 (root/"artifacts/ada-persistent-results.json").write_text(json.dumps(dict(summary=summary,results=results,resources=resources,
  region=8192,warmup_per_block=4,stable_per_run=args.samples,runs=args.runs),indent=2)+"\n")
 print(json.dumps(summary,indent=2),flush=True)
