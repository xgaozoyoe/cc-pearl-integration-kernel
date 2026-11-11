"""Isolated GPU0 warp-specialization experiment; no mining entry point."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse, ctypes as C, json, time, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
import cuda_capi as cc
P=C.c_void_p; I=C.c_int
lib=C.CDLL(os.environ.get("PEARL_WS_LIBRARY","/tmp/libpearl-ada-ws.so"))
lib.ws_run.argtypes=[I,P,P,I,I,I,I,P,P]
lib.ws_bench.argtypes=[I,P,P,I,I,I,I,P,I,P]
parser=argparse.ArgumentParser()
parser.add_argument("mode",choices=["validate","bench","profile","small"])
parser.add_argument("--variant",type=int,default=0)
parser.add_argument("--output",default="artifacts/ada-ws-results.json")
args=parser.parse_args()
def alloc(m,n,seed):
    rng=np.random.default_rng(seed)
    a=rng.integers(-128,128,(m,4096),dtype=np.int8)
    b=rng.integers(-128,128,(n,4096),dtype=np.int8)
    da,db,dt=cc.DBuf(a.nbytes),cc.DBuf(b.nbytes),cc.DBuf((m//16)*(n//16)*16*4)
    da.from_host(a);db.from_host(b)
    return da,db,dt
def run(v,a,b,t,m,n,info=None):
    cc._chk(lib.ws_run(v,a.ptr,b.ptr,m,n,4096,256,t.ptr,info),"ws_run")
def bench(v,a,b,t,m,n,iters):
    ms=C.c_float()
    cc._chk(lib.ws_bench(v,a.ptr,b.ptr,m,n,4096,256,t.ptr,iters,C.byref(ms)),"ws_bench")
    return ms.value
def telemetry():
    return subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit","--format=csv,noheader"],text=True).strip()
resources={}
for v in range(3):
    info=np.zeros(5,dtype=np.int32)
    cc._chk(lib.ws_run(v,None,None,256,192,4096,256,None,info.ctypes.data),"info")
    resources[str(v)]=dict(zip(["registers","threads","shared_bytes","active_ctas","local_bytes"],map(int,info)))
print("resources",json.dumps(resources),flush=True)
if args.mode in ("validate","small"):
    ref=C.CDLL("/tmp/libpearl-ada-linear-copy.so")
    ref.trial_fold.argtypes=[I,P,P,I,I,I,I,P]
    shapes=[(256,512,11),(512,256,23),(4096,4096,42),(256,768,57),(768,512,101)]
    if args.mode=="small":shapes=[(256,192,11)]
    results=[]
    for m,n,seed in shapes:
        a,b,t=alloc(m,n,seed)
        expected=np.empty((m//16)*(n//16)*16,dtype=np.uint32)
        cc._chk(ref.trial_fold(-1,a.ptr,b.ptr,m,n,4096,256,t.ptr),"DP4A")
        cc.sync();t.to_host(expected)
        for v in range(3):
            t.memset(0xA5)
            run(v,a,b,t,m,n);cc.sync()
            actual=np.empty_like(expected);t.to_host(actual)
            assert np.array_equal(actual,expected),(v,m,n,np.flatnonzero(actual!=expected)[:10].tolist())
            print(f"PASS v{v} {m}x{n} full transcript",flush=True)
            results.append(dict(variant=v,m=m,n=n,seed=seed,words=len(expected),pass_bitexact=True))
        for x in (a,b,t):x.free()
    if args.mode=="validate":
        (ROOT/"artifacts/ada-ws-correctness.json").write_text(json.dumps(results,indent=2)+"\n")
elif args.mode=="profile":
    a,b,t=alloc(4096,4096,42)
    for _ in range(40):run(args.variant,a,b,t,4096,4096);cc.sync()
else:
    m=n=4096
    a,b,t=alloc(m,n,42)
    for v in range(3):bench(v,a,b,t,m,n,2000)
    measurements=[]
    for repeat in range(2):
        for sample in range(40):
            # Rotate order to balance thermal drift; each sample averages 32 launches.
            for v in [(sample+repeat+i)%3 for i in range(3)]:
                start=time.perf_counter()
                ms=bench(v,a,b,t,m,n,32)
                wall_ms=(time.perf_counter()-start)*1000/32
                measurements.append(dict(run=repeat,variant=v,sample=sample,kernel_ms=ms,wall_ms=wall_ms))
            if sample%10==0:print(repeat,sample,telemetry(),flush=True)
    summaries={}
    for v in range(3):
        x=np.array([s["kernel_ms"] for s in measurements if s["variant"]==v])
        summaries[str(v)]=dict(median_ms=float(np.median(x)),mean_ms=float(np.mean(x)),p95_ms=float(np.percentile(x,95)),
            region_per_sec=1000/float(np.mean(x)),equivalent_THs=m*n*4096/(float(np.mean(x))*1e9),
            wall_mean_ms=float(np.mean([s["wall_ms"] for s in measurements if s["variant"]==v])))
    result=dict(dimensions=[m,n,4096],samples_per_run=40,runs=2,kernels_per_sample=32,resources=resources,
                summaries=summaries,measurements=measurements,telemetry_end=telemetry())
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(summaries,indent=2))
