"""GPU0-only diagnostic; repeated K64 inputs, not a valid mining workload."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import ctypes as C,json,sys,time,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
import cuda_capi as cc
P=C.c_void_p;I=C.c_int
lib=C.CDLL('/tmp/libpearl-rank128-layers.so')
lib.layers.argtypes=[I,P,P,I,I,P,P]
lib.layers_time.argtypes=[I,P,P,I,I,P,I,P]
lib.roof_run.argtypes=[P,P,P,P,I,I,P,P]
resources={}
for v in (0,1):
 info=np.zeros(5,np.int32);cc._chk(lib.layers(v,None,None,256,192,None,info.ctypes.data),'info')
 assert info[4]==0 and info[3]==1,info
 resources[str(v)]=info.tolist()
info=np.zeros(4,np.int32);cc._chk(lib.roof_run(None,None,None,None,1,1,info.ctypes.data,None),'roof info')
assert info[1]==0 and info[3]==1,info
resources['roof']=info.tolist();print('resources',resources,flush=True)
rng=np.random.default_rng(1028)
# Verify every lane's sum of its 128 native MMA accumulators, signed input.
validation=[]
for seed in (42,1234):
 rng=np.random.default_rng(seed)
 ah=rng.integers(-16,16,(256,64),np.int8);bh=rng.integers(-16,16,(192,64),np.int8)
 a=cc.DBuf(256*4096);b=cc.DBuf(192*4096);out=cc.DBuf(384*4)
 a.from_host(np.tile(ah,(1,64)));b.from_host(np.tile(bh,(1,64)))
 ref=ah.astype(np.int64)@bh.astype(np.int64).T*64
 expected=[]
 for tid in range(384):
  warp,lane=divmod(tid,32);wm=warp%4;wn=warp//4
  rows=[wm*64+r*16+lane//4+dr for r in range(4) for dr in (0,8)]
  cols=[wn*64+g*16+h*8+2*(lane%4)+dc for g in range(4) for h in range(2) for dc in (0,1)]
  expected.append(int(ref[np.ix_(rows,cols)].sum())&0xffffffff)
 for v in (0,1):
  cc._chk(lib.layers(v,a.ptr,b.ptr,256,192,out.ptr,None),'validate')
  got=np.empty(384,np.uint32);out.to_host(got)
  assert np.array_equal(got,np.array(expected,np.uint32)),(seed,v,got[:8],expected[:8])
  validation.append(dict(seed=seed,variant=v,passed=True))
print('CPU accumulator validation PASS',flush=True)
if '--validate-only' in sys.argv:sys.exit(0)
# 8192 region matches common mining region; identical data for both layers.
m=n=8192;blocks=((m+255)//256)*((n+191)//192)
a=cc.DBuf(m*4096);b=cc.DBuf(n*4096);out=cc.DBuf(blocks*384*4)
a.from_host(np.tile(rng.integers(-16,16,(m,64),np.int8),(1,64)))
b.from_host(np.tile(rng.integers(-16,16,(n,64),np.int8),(1,64)))
inp=cc.DBuf(384*28*4);tim=cc.DBuf(blocks*12*2*8);sm=cc.DBuf(blocks*4)
inp.from_host(rng.integers(0,2**32,(384,28),dtype=np.uint32))
def run(v):
 ms=C.c_float();start=time.perf_counter()
 if v==2:rc=lib.roof_run(inp.ptr,out.ptr,tim.ptr,sm.ptr,blocks,64,None,C.byref(ms))
 else:rc=lib.layers_time(v,a.ptr,b.ptr,m,n,out.ptr,1,C.byref(ms))
 cc._chk(rc,'run');return ms.value,(time.perf_counter()-start)*1000
# Full-region bit-exact comparison also exercises CTA permutation and edge padding.
run(0);x=np.empty(blocks*384,np.uint32);out.to_host(x)
run(1);y=np.empty_like(x);out.to_host(y);assert np.array_equal(x,y)
print('8192 region full/shared equality PASS',flush=True)
start=time.perf_counter()
while time.perf_counter()-start<5:
 for v in (0,1,2):run(v)
rows=[];telemetry=[]
for repeat in range(2):
 for sample in range(40):
  for j in range(3):
   v=(sample+j+repeat)%3;ms,wall=run(v)
   rows.append(dict(run=repeat,sample=sample,variant=v,ms=ms,wall_ms=wall))
  if sample%20==0:
   t=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip()
   telemetry.append(t);print(repeat,sample,t,flush=True)
summary={}
for v,name in enumerate(('full_copy','shared_resident','register_roof')):
 vals=np.array([r['ms'] for r in rows if r['variant']==v]);wall=[r['wall_ms'] for r in rows if r['variant']==v]
 summary[name]=dict(mean_ms=float(vals.mean()),median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),mean_host_ms=float(np.mean(wall)),executed_TMACs=blocks*12*64*64*4096/(vals.mean()*1e9),run_means=[float(np.mean([r['ms'] for r in rows if r['variant']==v and r['run']==q])) for q in (0,1)])
result=dict(summary=summary,resources=resources,validation=validation,full_region_equality=True,blocks=blocks,telemetry=telemetry,measurements=rows,note='Diagnostic repeated K64; no transcript. Rates count padded executed MACs, not mining hashrate. Roof is a separate calibration with a different operand schedule. Shared initialization remains inside event timing.')
(root/'artifacts/rank128-layers-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
