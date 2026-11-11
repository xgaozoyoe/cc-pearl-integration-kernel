"""Matched short-region roof vs existing v121; GPU0, offline only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import ctypes as C,json,sys,time,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"python"))
import cuda_capi as cc
P=C.c_void_p;I=C.c_int
roof=C.CDLL("/tmp/libpearl-imma-roof.so")
roof.roof_run.argtypes=[P,P,P,P,I,I,P,P]
base=C.CDLL("/tmp/libpearl-ada-ws.so")
base.ws_bench.argtypes=[I,P,P,I,I,I,I,P,I,P]
rng=np.random.default_rng(42)
a=cc.DBuf(4096*4096);b=cc.DBuf(4096*4096);t=cc.DBuf(4096*4096//256*16*4)
a.from_host(rng.integers(-128,128,(4096,4096),np.int8))
b.from_host(rng.integers(-128,128,(4096,4096),np.int8))
inp=cc.DBuf(384*28*4);out=cc.DBuf(352*384*4);tim=cc.DBuf(352*12*2*8);sm=cc.DBuf(352*4)
inp.from_host(rng.integers(0,2**32,(384,28),dtype=np.uint32))
def run(v):
 ms=C.c_float()
 if v=="baseline":rc=base.ws_bench(0,a.ptr,b.ptr,4096,4096,4096,256,t.ptr,1,C.byref(ms))
 else:rc=roof.roof_run(inp.ptr,out.ptr,tim.ptr,sm.ptr,352,64,None,C.byref(ms))
 cc._chk(rc,v);return ms.value
start=time.perf_counter()
while time.perf_counter()-start<3:
 run("baseline");run("register_roof")
rows=[];telemetry=[]
for repeat in range(2):
 for sample in range(40):
  order=["baseline","register_roof"]
  if (sample+repeat)%2:order.reverse()
  for v in order:rows.append(dict(run=repeat,sample=sample,variant=v,ms=run(v)))
  if sample%20==0:
   telemetry.append(subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit","--format=csv,noheader"],text=True).strip())
summary={}
for v in ("baseline","register_roof"):
 vals=np.array([r["ms"] for r in rows if r["variant"]==v])
 summary[v]=dict(median_ms=float(np.median(vals)),mean_ms=float(vals.mean()),p95_ms=float(np.percentile(vals,95)),
  useful_work_equiv_THs=4096**3/(float(np.median(vals))*1e9))
result=dict(summaries=summary,telemetry=telemetry,measurements=rows,
 note="Both 352 CTA x12 warps x64 K64 iterations; roof operands repeat, no memory work in loop. Useful equivalent applies original region padding convention; not mining throughput.")
(root/"artifacts/ada-imma-roof-matched.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result["summaries"],indent=2))
print(telemetry)
