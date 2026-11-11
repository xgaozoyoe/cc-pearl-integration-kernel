"""GPU0 only, register-resident IMMA ceiling; not a mining benchmark."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import ctypes as C,json,sys,time,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"python"))
import cuda_capi as cc
P=C.c_void_p;I=C.c_int
lib=C.CDLL("/tmp/libpearl-imma-roof.so")
lib.roof_run.argtypes=[P,P,P,P,I,I,P,P]
info=np.zeros(4,np.int32)
cc._chk(lib.roof_run(None,None,None,None,1,1,info.ctypes.data,None),"info")
assert info[1]==0 and info[3]==1,info
print("resources",info.tolist(),flush=True)
inp=cc.DBuf(384*28*4);out=cc.DBuf(352*384*4)
tim=cc.DBuf(352*12*2*8);sm=cc.DBuf(352*4)
def run(blocks,iters):
    ms=C.c_float()
    cc._chk(lib.roof_run(inp.ptr,out.ptr,tim.ptr,sm.ptr,blocks,iters,None,C.byref(ms)),"roof")
    return ms.value
def telem():
    return subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu","--format=csv,noheader"],text=True).strip()
validation=[]
for vals in [(1,2,3),(-1,2,-3)]:
    host=np.empty((384,28),np.uint32)
    for row in range(4):host[:,4*row:4*row+4]=np.uint32((row+1)*0x01010101)
    for slot,val in enumerate(vals):host[:,16+4*slot:20+4*slot]=np.uint32((val&255)*0x01010101)
    inp.from_host(host)
    for iters in (1,64,4096):
        run(128,iters)
        got=np.empty(352*384,np.uint32);out.to_host(got)
        expected=(256*iters*10*sum(vals[g%3]+vals[(g+4)%3] for g in range(4)))&0xffffffff
        assert np.all(got[:128*384]==expected),(vals,iters,expected,got[:10])
        validation.append(dict(values=vals,iterations=iters,expected=expected,passed=True))
rng=np.random.default_rng(123)
inp.from_host(rng.integers(0,2**32,(384,28),dtype=np.uint32))
cases=[(128,256),(128,1024),(128,4096),(352,64)]
# Keep GPU busy for >=3 seconds before measurements.
start=time.perf_counter()
while time.perf_counter()-start<3:run(128,4096)
rows=[];telemetry=[];cycle_stats={}
for repeat in range(2):
    for sample in range(40):
        for j in range(len(cases)):
            blocks,iters=cases[(sample+repeat+j)%len(cases)]
            ms=run(blocks,iters)
            rows.append(dict(run=repeat,sample=sample,blocks=blocks,iterations=iters,ms=ms))
        if sample%20==0:
            t=telem();telemetry.append(t);print(repeat,sample,t,flush=True)
for blocks,iters in cases:
    run(blocks,iters)
    ht=np.empty((352,12,2),np.uint64);hs=np.empty(352,np.uint32)
    tim.to_host(ht);sm.to_host(hs)
    spans=[]
    for smid in np.unique(hs[:blocks]):
        indices=np.flatnonzero(hs[:blocks]==smid)
        span=int(ht[indices,:,1].max())-int(ht[indices,:,0].min())
        mma=len(indices)*12*64*iters
        spans.append(dict(sm=int(smid),ctas=len(indices),cycles=span,mma_per_cycle=mma/span))
    cycle_stats[f"{blocks}x{iters}"]=dict(active_sms=len(spans),sms=spans,
        mma_per_sm_cycle_mean=float(np.mean([s["mma_per_cycle"] for s in spans])))
summary={}
for blocks,iters in cases:
    vals=np.array([r["ms"] for r in rows if r["blocks"]==blocks and r["iterations"]==iters])
    mma=blocks*12*64*iters
    summary[f"{blocks}x{iters}"]=dict(median_ms=float(np.median(vals)),mean_ms=float(vals.mean()),
        p95_ms=float(np.percentile(vals,95)),warp_mma_count=mma,
        executed_TMAC_per_s=mma*4096/(float(np.median(vals))*1e9))
result=dict(resources=dict(zip(["registers","local_bytes","reserved_shared_bytes","ctas_per_sm"],map(int,info))),
    correctness=validation,telemetry=telemetry,summaries=summary,cycles=cycle_stats,measurements=rows)
(root/"artifacts/ada-imma-roof-results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(summary,indent=2),flush=True)
for name,v in cycle_stats.items():print(name,v["active_sms"],v["mma_per_sm_cycle_mean"],flush=True)
