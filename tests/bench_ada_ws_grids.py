"""Offline mainnet-sized WS test. Production generation; no jackpot or pool."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys,ctypes as C,json,time,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
import cuda_capi as cc
import miner_capi as miner
from pool_common import M,N,K,R,real_config
import pearl_host
P=C.c_void_p; I=C.c_int
lib=C.CDLL("/tmp/libpearl-ada-ws.so")
lib.ws_run.argtypes=[I,P,P,I,I,I,I,P,P]
rt=C.CDLL("libcudart.so.12")
rt.cudaEventCreate.argtypes=[C.POINTER(P)]
rt.cudaEventRecord.argtypes=[P,P]
rt.cudaEventElapsedTime.argtypes=[C.POINTER(C.c_float),P,P]
rt.cudaEventDestroy.argtypes=[P]
def event():
    e=P();cc._chk(rt.cudaEventCreate(C.byref(e)),"event create");return e
events=[(event(),event()) for _ in range(1024)]
buf=miner.Bufs(4096)
key=pearl_host.derive_key(bytes(80),real_config())
buf.dkeyjob.from_host(np.frombuffer(key,np.uint8).copy())
def grid(v,seed):
    start=time.perf_counter()
    cc.setup_job(buf.dA,buf.dB,buf.dBt_full,buf.dkeyjob,buf.dnsA,buf.dnsB,M,N,K,R,seed)
    cc.noise_gen(buf.dEAL,buf.dEAR,buf.dEBL,buf.dEBR,buf.dnsA,buf.dnsB,M,N,K,R)
    cc.transpose_i8(buf.dEAR,buf.dEAR_t,R,K,K,0)
    cc.sync()
    idx=0
    for r0 in range(0,M,4096):
        cc.noise_gemm(buf.dEAL.offset(r0*R),buf.dEAR_t,buf.dA.offset(r0*K),buf.dApEA,4096,K,R)
        for j,c0 in enumerate(range(0,N,4096)):
            if r0==0:cc.noise_gemm(buf.dEBR.offset(c0*R),buf.dEBL,buf.dBt_full.offset(c0*K),buf.dBpEB[j],4096,K,R)
            begin,end=events[idx];idx+=1
            cc._chk(rt.cudaEventRecord(begin,None),"record")
            cc._chk(lib.ws_run(v,buf.dApEA.ptr,buf.dBpEB[j].ptr,4096,4096,K,R,buf.dtb.ptr,None),"ws")
            cc._chk(rt.cudaEventRecord(end,None),"record")
    cc.sync()
    wall_ms=1000*(time.perf_counter()-start)
    kernel_ms=0
    for begin,end in events:
        ms=C.c_float();cc._chk(rt.cudaEventElapsedTime(C.byref(ms),begin,end),"elapsed");kernel_ms+=ms.value
    return dict(kernel_ms=kernel_ms,wall_ms=wall_ms)
for v in range(3):
    for i in range(2):grid(v,0x504541524c000000+i)
rows=[]
for repeat in range(2):
    for sample in range(40):
        for v in [(sample+repeat+i)%3 for i in range(3)]:
            result=grid(v,0x504541524c000100+repeat*40+sample)
            rows.append(dict(run=repeat,sample=sample,variant=v,**result))
        if sample%5==0:
            status=subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit","--format=csv,noheader"],text=True).strip()
            print(repeat,sample,status,rows[-3:],flush=True)
summaries={}
for v in range(3):
    subset=[r for r in rows if r["variant"]==v]
    def stats(key):
        x=np.array([r[key] for r in subset])
        return dict(median_ms=float(np.median(x)),mean_ms=float(np.mean(x)),p95_ms=float(np.percentile(x,95)),
                    grids_per_second=1000/float(np.mean(x)),equivalent_THs=M*N*K/(float(np.mean(x))*1e9))
    summaries[str(v)]=dict(kernel=stats("kernel_ms"),offline_grid_wall=stats("wall_ms"))
result=dict(dimensions=[M,N,K],region=4096,runs=2,samples_per_run=40,
            note="Full transcript and production A/B/V3/noise generation; PoW/jackpot/pool omitted in all variants. Kernel events summed over 1024 launches; wall includes setup/noise/dispatch.",
            summaries=summaries,measurements=rows)
(ROOT/"artifacts/ada-ws-grid-results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(summaries,indent=2),flush=True)
for pair in events:
    for e in pair:rt.cudaEventDestroy(e)
