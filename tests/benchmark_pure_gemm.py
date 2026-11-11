"""Controlled existing-Ada-kernel comparison. Offline, GPU 0, no shares.

Both modes use the same setup/noise path and identical seeds. Both omit PoW
BLAKE3/checks to isolate transcript cost. Separate event-instrumented runs keep
event-recording overhead out of the primary full-grid wall measurements.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import ctypes,json,re,sys,time,statistics,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import cuda_capi as cc
import miner_capi as miner
from pool_common import real_config
P=ctypes.c_void_p;I=ctypes.c_int
lib=ctypes.CDLL('/tmp/libpearl-pure-gemm.so')
lib.trial_fold.argtypes=[I,P,P,I,I,I,I,P]
lib.trial_info.argtypes=[I,P]
lib.trial_pow.argtypes=[I,P,P,I,I,I,I,P,P,P,P,P,P]
lib.trial_operator.argtypes=[I,P,P,I,I,I,I,P]
lib.measure_init.argtypes=[I];lib.measure_end.argtypes=[P]
lib.benchmark_device_info.argtypes=[P];lib.benchmark_lane_map.argtypes=[P]
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
names={20:'full transcript',48:'pure GEMM'}

def verify():
    coords=np.empty((256,4,8,4,2),np.int32)
    cc._chk(lib.benchmark_lane_map(coords.ctypes.data_as(P)),'lane map')
    for tid in range(256):
        warp,lane=divmod(tid,32)
        for tm in range(4):
            for tn in range(8):
                for v in range(4):
                    expected=(16*(warp%2)+32*tm+lane//4+8*(v//2),
                              16*(warp//2)+64*(tn//2)+8*(tn%2)+2*(lane%4)+v%2)
                    assert tuple(coords[tid,tm,tn,v])==expected
    np.save(OUT/'pure-gemm-lane-map.npy',coords)
    for m,n,seed in [(256,512,11),(512,256,23),(4096,4096,42)]:
        rng=np.random.default_rng(seed)
        a=cc.DBuf(m*4096);b=cc.DBuf(n*4096);t=cc.DBuf((m//16)*(n//16)*64)
        try:
            a.from_host(rng.integers(-128,128,(m,4096),dtype=np.int8))
            b.from_host(rng.integers(-128,128,(n,4096),dtype=np.int8))
            ref=np.empty((m//16)*(n//16)*16,np.uint32)
            cc._chk(lib.trial_fold(-1,a.ptr,b.ptr,m,n,4096,256,t.ptr),'DP4A')
            cc.sync();t.to_host(ref)
            cc._chk(lib.trial_fold(20,a.ptr,b.ptr,m,n,4096,256,t.ptr),'full')
            actual=np.empty_like(ref);cc.sync();t.to_host(actual)
            assert np.array_equal(actual,ref)
            t.memset(0xA5)
            cc._chk(lib.trial_fold(48,a.ptr,b.ptr,m,n,4096,256,t.ptr),'pure')
            cc.sync();t.to_host(actual)
            checksum=actual[:(m//128)*(n//256)*8].reshape(n//256,m//128,8)
            ref=ref.reshape(m//16,n//16,16)
            for bn in range(n//256):
                for bm in range(m//128):
                    for warp in range(8):
                        words=[ref[bm*8+warp%2+2*tm,bn*16+warp//2+4*tn,15]
                               for tm in range(4) for tn in range(4)]
                        assert checksum[bn,bm,warp]==np.bitwise_xor.reduce(words)
            assert np.all(actual[checksum.size:]==np.uint32(0xA5A5A5A5))
            assert lib.trial_pow(48,None,None,m,n,4096,256,None,None,None,None,None,None)!=0
            print(f'PASS reference transcript, per-warp GEMM checksum, no transcript writes: {m}x{n}',flush=True)
        finally:
            for buf in (a,b,t):buf.free()

def summary(values):
    med=float(np.median(values))
    return {'samples_ms':values,'median_ms':med,'mean_ms':float(np.mean(values)),
            'p95_ms':float(np.percentile(values,95)),
            'grids_per_second':1000/med,'equivalent_THS':70.368744177664/(med/1000)}

def main():
    verify()
    dev=np.empty(4,np.int32);cc._chk(lib.benchmark_device_info(dev.ctypes.data_as(P)),'device')
    resources={}
    for v in names:
        info=np.empty(5,np.int32);cc._chk(lib.trial_info(v,info.ctypes.data_as(P)),'info')
        assert info[3]>0
        resources[str(v)]={'registers_per_thread':int(info[0]),'threads_per_CTA':int(info[1]),
            'shared_bytes_per_CTA':int(info[2]),'resident_CTAs_per_SM':int(info[3]),
            'local_bytes_per_thread':int(info[4]),'warps_per_CTA':int(info[1]//32),
            'theoretical_warp_occupancy_pct':100*info[3]*(info[1]//32)/dev[1]}
    cc._chk(lib.measure_init(1024),'events init')
    original_setup=cc.setup_job;original_pow=cc.pearl_pow_split
    buffers=miner.Bufs(4096)
    results={'resources':resources,'device':dev.tolist(),'measurements':{},'blocks':[],
             'warmup_grids_per_block':4,'stable_grids_per_mode_per_phase':20,
             'matrix_generation':'unchanged GPU setup_job, identical deterministic seeds across modes',
             'comparison':'both omit PoW BLAKE3/jackpot checks; full retains all transcript operations'}
    telemetry=open(OUT/'pure-gemm-telemetry.csv','w')
    monitor=subprocess.Popen(['nvidia-smi','-i','0',
        '--query-gpu=timestamp,index,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,temperature.gpu,pstate',
        '--format=csv,noheader,nounits','-lms','200'],stdout=telemetry,stderr=subprocess.DEVNULL)
    class Pool:
        def check_newer_job(self,job_id):return None
    class Scheduler:
        mode='user'
        def __init__(self):self.grids=0
        def note(self,elapsed):pass
        def maybe_switch(self):self.grids+=1;return self.grids>=14
    try:
        for instrumented in (False,True):
            phase='events' if instrumented else 'wall'
            counts={20:0,48:0};collected={20:[],48:[]};kernel={20:[],48:[]}
            for v in (20,48,48,20):
                started=time.time();grid_index=0
                def setup(*args):
                    nonlocal grid_index
                    seed=0x504541524C000000+counts[v]
                    counts[v]+=1;grid_index+=1
                    if instrumented:cc._chk(lib.measure_begin(),'events begin')
                    return original_setup(*args[:-1],seed)
                def operator(a,b,m,n,k,r,key,target,tb,dig,found,coord,variant):
                    cc._chk(lib.trial_operator(v,cc._as(a),cc._as(b),m,n,k,r,cc._as(tb)),'operator')
                def log(message):
                    match=re.search(r'grid wall: ([\d.]+) s',message)
                    if not match:return
                    wall=float(match[1])*1000;km=None
                    if instrumented:
                        ms=ctypes.c_float();cc._chk(lib.measure_end(ctypes.byref(ms)),'events end');km=ms.value
                    if grid_index>4:
                        collected[v].append(wall)
                        if km is not None:kernel[v].append(km)
                    print(f'{phase} {names[v]} grid={grid_index} wall={wall:.2f} ms'+
                          (f' kernel={km:.3f} ms' if km is not None else '')+
                          (' warmup' if grid_index<=4 else ''),flush=True)
                cc.setup_job=setup;cc.pearl_pow_split=operator
                miner.mine_job(Pool(),real_config(),bytes(80),0,'pure-gemm-offline',4096,0,
                    Scheduler(),buffers,{'user':0,'dev':0},None,log)
                results['blocks'].append({'phase':phase,'variant':v,'start_unix':started,'end_unix':time.time()})
            for v in names:
                assert len(collected[v])==20
            results['measurements'][phase]={'wall':{str(v):summary(collected[v]) for v in names}}
            if instrumented:results['measurements'][phase]['kernel']={str(v):summary(kernel[v]) for v in names}
        for label,rows in [('wall',results['measurements']['wall']['wall']),
                           ('kernel',results['measurements']['events']['kernel'])]:
            full,pure=rows['20']['median_ms'],rows['48']['median_ms']
            results[label+'_comparison']={'overhead_ms':full-pure,'overhead_pct_of_full':100*(full-pure)/full,
                'slowdown_ratio':full/pure,'slowdown_pct_vs_pure':100*(full/pure-1)}
        (OUT/'pure-gemm-results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2),flush=True)
    finally:
        monitor.terminate();monitor.wait();telemetry.close()
        cc.setup_job=original_setup;cc.pearl_pow_split=original_pow
        lib.measure_close()
        for value in vars(buffers).values():
            if isinstance(value,cc.DBuf):value.free()
            elif isinstance(value,list):
                for buf in value:buf.free()
if __name__=='__main__':main()
