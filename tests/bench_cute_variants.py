"""Isolated GPU-0 transcript correctness and CTA tuning benchmark.
Uses /tmp/libpearl-cute-variants.so; leaves libp40cuda.so unchanged.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys, ctypes, statistics, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
import cuda_capi as cc

lib=ctypes.CDLL(os.environ.get('PEARL_TRIAL_LIBRARY','/tmp/libpearl-cute-variants.so'))
RESULT_PREFIX=os.environ.get('PEARL_RESULT_PREFIX','/tmp/pearl-cute')
P=ctypes.c_void_p; I=ctypes.c_int
lib.trial_fold.argtypes=[I,P,P,I,I,I,I,P]
lib.trial_info.argtypes=[I,P]
lib.trial_bench.argtypes=[I,P,P,I,I,I,I,P,I,P]
lib.trial_pow.argtypes=[I,P,P,I,I,I,I,P,P,P,P,P,P]
NAMES=['128x256 s3 baseline','128x128 s3','64x256 s3','64x128 s3',
       '128x128 s4','128x256 s2','128x256 s3 shared100',
       '128x128 s3 shared100','64x256 s3 shared100','64x256 s4 shared100',
       '128x256 s4','128x256 s5','256x128 s3','256x128 s4','128x128 s2','128x256 s3 cache-L1','128x256 BK64 s2','128x256 BK64 s3']

if os.environ.get('PEARL_TRIAL_NAMES'):
    NAMES=json.loads(os.environ['PEARL_TRIAL_NAMES'])

def gate(m,n,seed):
    k=4096;r=256
    rng=np.random.default_rng(seed)
    a=cc.DBuf(m*k);b=cc.DBuf(n*k);t=cc.DBuf((m//16)*(n//16)*64)
    passed=set()
    try:
        a.from_host(rng.integers(-128,128,(m,k),dtype=np.int8))
        b.from_host(rng.integers(-128,128,(n,k),dtype=np.int8))
        ref=np.empty((m//16)*(n//16)*16,np.uint32)
        cc._chk(lib.trial_fold(-1,a.ptr,b.ptr,m,n,k,r,t.ptr),'DP4A reference')
        cc.sync();t.to_host(ref)
        out=np.empty_like(ref)
        for v,name in enumerate(NAMES):
            t.memset(0xA5)
            cc._chk(lib.trial_fold(v,a.ptr,b.ptr,m,n,k,r,t.ptr),name)
            cc.sync();t.to_host(out)
            if np.array_equal(ref,out):
                passed.add(v)
            else:
                print(f'FAIL excluded {v}: {name}, transcript mismatch, shape={m,n}, seed={seed}',flush=True)
        print(f'PASS {len(passed)}/{len(NAMES)} variants vs DP4A: M={m} N={n} K={k} R={r} seed={seed}',flush=True)
        return passed
    finally:
        for buf in (a,b,t):buf.free()

def bench(valid):
    m=n=k=4096;r=256
    rng=np.random.default_rng(100)
    a=cc.DBuf(m*k);b=cc.DBuf(n*k);t=cc.DBuf((m//16)*(n//16)*64)
    a.from_host(rng.integers(-128,128,(m,k),dtype=np.int8))
    b.from_host(rng.integers(-128,128,(n,k),dtype=np.int8))
    results={}
    try:
        # Several passes in reverse/forward order reduce clock-ramp/order bias.
        for order in (sorted(valid),sorted(valid,reverse=True),sorted(valid)):
            for v in order:
                info=np.empty(5,np.int32)
                cc._chk(lib.trial_info(v,info.ctypes.data_as(P)),'info')
                ms=ctypes.c_float()
                cc._chk(lib.trial_bench(v,a.ptr,b.ptr,m,n,k,r,t.ptr,10,ctypes.byref(ms)),'warmup')
                cc._chk(lib.trial_bench(v,a.ptr,b.ptr,m,n,k,r,t.ptr,100,ctypes.byref(ms)),'timing')
                entry=results.setdefault(v,{'name':NAMES[v],'resources':info.tolist(),'ms':[]})
                entry['ms'].append(ms.value)
        base=statistics.median(results[0]['ms'])
        for v in results:
            x=results[v];x['median_ms']=statistics.median(x['ms']);x['speedup']=base/x['median_ms']
        for v,x in sorted(results.items(),key=lambda p:p[1]['median_ms']):
            print(f"{v}: {x['name']:28s} {x['median_ms']:.4f}ms {x['speedup']:.3f}x resources(reg,threads,smem,blocks,local)={x['resources']} samples={x['ms']}",flush=True)
        Path(RESULT_PREFIX+'-variants-results.json').write_text(json.dumps(results,indent=2))
    finally:
        for buf in (a,b,t):buf.free()

if __name__=='__main__':
    valid=gate(256,512,11) & gate(512,256,23) & gate(4096,4096,42)
    assert 0 in valid, 'Baseline must pass DP4A correctness'
    Path(RESULT_PREFIX+'-correctness-results.json').write_text(json.dumps({'passed':sorted(valid),'failed':sorted(set(range(len(NAMES)))-valid)},indent=2))
    bench(valid)
