"""GPU-0 diagnostic ablation, not a mining/hashrate benchmark.

Final-only modes must match the final word of the complete DP4A transcript.
All accumulator elements remain live through the final XOR; neither GEMM nor
intermediate K work can be optimized away. Local-only modes are deliberately
invalid transcripts and are never passed to trial_pow.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys, ctypes, json, statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
import cuda_capi as cc
lib=ctypes.CDLL(os.environ.get('PEARL_TRIAL_LIBRARY','/tmp/libpearl-fold-ablation.so'))
P=ctypes.c_void_p; I=ctypes.c_int
lib.trial_fold.argtypes=[I,P,P,I,I,I,I,P]
lib.trial_info.argtypes=[I,P]
lib.trial_bench.argtypes=[I,P,P,I,I,I,I,P,I,P]
lib.trial_pow.argtypes=[I,P,P,I,I,I,I,P,P,P,P,P,P]
names={11:'8x1 full transcript',20:'2x4 full transcript',
       23:'8x1 final-only GEMM',24:'2x4 final-only GEMM',
       25:'8x1 local XOR only',26:'2x4 local XOR only'}
occupancy = len(sys.argv)>1 and sys.argv[1]=='occupancy'
cache_trial = len(sys.argv)>1 and sys.argv[1]=='l1'
square_trial = len(sys.argv)>1 and sys.argv[1]=='square'
mainloop_trial = len(sys.argv)>1 and sys.argv[1]=='mainloop'
reuse_trial = len(sys.argv)>1 and sys.argv[1]=='reuse'
mapping_trial = len(sys.argv)>1 and sys.argv[1]=='mapping'
operand_trial = len(sys.argv)>1 and sys.argv[1]=='operand'
prefetch_trial = len(sys.argv)>1 and sys.argv[1] in ('prefetch','prefetch2','ldsm','ldsm-order','rhythm','ring','ada-v2','ada-layout','ada-bpack','ada-copy','ada-swizzle')
if occupancy:
    names={11:'8x1 full transcript',20:'2x4 full transcript',
           27:'128x128 BK32 warp2x4',28:'128x128 BK64 warp2x4',
           29:'64x256 BK32 warp1x8',30:'128x256 BK32 warp2x4'}
if cache_trial:
    names={20:'best 2x4 CG BK64',27:'small CG BK32',
           31:'small CA BK32',32:'small CA pair132',33:'small CA pair128',
           34:'small CA BK64 pair132',35:'large CA BK32',36:'large CA BK64',
           37:'small CG pair132',38:'small CA s4 pair132'}
if square_trial:
    names={20:'best 2x4 CG BK64',39:'192x192 12warp 3x4 BK32 s3',
           40:'192x192 12warp 3x4 BK64 s3',41:'192x192 12warp 3x4 BK32 s4',
           42:'192x192 12warp 4x3 BK32 s3',43:'192x192 8warp BK32 s3',
           44:'192x192 8warp BK32 s4',45:'192x192 8warp BK64 shared transcript',
           46:'192x192 8warp BK32 shared transcript',47:'192x192 8warp 2x4 BK32 s3'}
if mainloop_trial:
    names={20:'baseline K64',49:'prefetch before LDSM',50:'row-major serpentine MMA',
           51:'row-major linear MMA',52:'consume K32 fragments separately',53:'B LDSM x4',
           54:'K128 s2 consume K32 separately',55:'K128 s2 full fragment load',
           56:'K32 lookahead joint A/B',57:'K32 lookahead staggered A/B'}
if reuse_trial:
    names={20:'baseline',58:'group M1',59:'group M4',60:'group M8',61:'group M16',
           62:'group M2',63:'group M12',64:'group M24',65:'group N2',66:'group N4',67:'group N8'}
if mapping_trial:
    names={67:'best 128x256 warp2x4',68:'256x128 warp4x2',69:'256x128 warp2x4',
           70:'256x128 warp8x1',71:'128x128 warp2x2',72:'128x128 warp1x4',73:'128x128 warp4x1',
           74:'128x320 register transcript',75:'128x320 direct transcript',
           76:'128x320 shared transcript s3',77:'128x320 shared transcript s4'}
if operand_trial:
    names={67:'baseline',78:'wide stream B shared transcript',79:'wide stream A/B shared transcript',
           80:'wide stream B register transcript',81:'baseline stream B',
           82:'256x192 stream B s3',83:'256x192 stream A/B s3',84:'256x192 register transcript',85:'256x192 stream B s4',
           86:'256x192 K64 stream B s3',87:'256x192 K64 stream A/B s3',88:'256x192 K64 stream B s2'}
if prefetch_trial:
    names={67:'N8 baseline',88:'large CTA baseline',89:'B prefetch before group',
           90:'B prefetch after first half',91:'B prefetch midway first half'}
    if sys.argv[1]=='prefetch2':
        names={89:'B prefetch baseline',92:'B prefetch s3',93:'dead-row A prefetch s2',94:'dead-row A prefetch s3'}
    if sys.argv[1]=='ldsm':
        names={89:'B prefetch baseline',95:'LDSM after two MMA rows',96:'LDSM after three MMA rows'}
    if sys.argv[1]=='ldsm-order':
        names={89:'B prefetch baseline',97:'rotating MMA rows',98:'interleaved MMA halves'}
    if sys.argv[1]=='rhythm':
        names={89:'B prefetch baseline',99:'K64 rolling B / bulk A',100:'K64 rolling B / dead-row A'}
    if sys.argv[1]=='ring':
        names={89:'B prefetch baseline',101:'three-slot B ring',102:'two-slot B ring'}
    if sys.argv[1]=='ada-v2':
        names={89:'B prefetch baseline',103:'hand Ada v2 lead1',104:'hand Ada v2 lead2',
               105:'hand Ada v2 streamed B',106:'hand Ada v2 K32 prefetch'}
    if sys.argv[1]=='ada-layout':
        names={109:'raw address regrouping',107:'K64 planar K32',108:'K96 planar K32'}
    if sys.argv[1]=='ada-bpack':
        names={109:'raw address regrouping',110:'MMA-packed B x4'}
if prefetch_trial and sys.argv[1]=='ada-copy':
    names={110:'packed B fixed K baseline',111:'unrolled async copies',112:'fixed async copy rounds',113:'packed B three stages',114:'fixed copies three stages',115:'warp 64x96 operand reuse'}
if prefetch_trial and sys.argv[1]=='ada-swizzle':
    names={110:'baseline xor2 shift3',116:'no swizzle',117:'xor3 shift3',118:'xor2 shift2',119:'xor3 shift2',120:'ordered async producer lanes',121:'ordered producer direct shared address'}
full_variants = occupancy or cache_trial or square_trial or mainloop_trial or reuse_trial or mapping_trial or operand_trial or prefetch_trial
def gate(m,n,seed):
    rng=np.random.default_rng(seed); k=4096; r=256
    a=cc.DBuf(m*k); b=cc.DBuf(n*k); t=cc.DBuf((m//16)*(n//16)*64)
    passed=set()
    try:
        a.from_host(rng.integers(-128,128,(m,k),dtype=np.int8))
        b.from_host(rng.integers(-128,128,(n,k),dtype=np.int8))
        ref=np.empty((m//16)*(n//16)*16,np.uint32)
        cc._chk(lib.trial_fold(-1,a.ptr,b.ptr,m,n,k,r,t.ptr),'reference')
        cc.sync(); t.to_host(ref)
        for v in (list(names) if full_variants else (11,20,23,24)):
            info=np.empty(5,np.int32)
            cc._chk(lib.trial_info(v,info.ctypes.data_as(P)),'gate info')
            if info[3]==0:
                print(f'FAIL excluded {v}: cannot launch, resources={info.tolist()}',flush=True)
                continue
            t.memset(0); out=np.empty_like(ref)
            cc._chk(lib.trial_fold(v,a.ptr,b.ptr,m,n,k,r,t.ptr),'gate')
            cc.sync(); t.to_host(out)
            if v not in (23,24):
                if not np.array_equal(out,ref):
                    assert v not in (11,20), names[v]
                    print(f'FAIL excluded {v}: {names[v]}, shape={m,n}',flush=True)
                    continue
            else:
                assert np.array_equal(out.reshape(-1,16)[:,15],ref.reshape(-1,16)[:,15]), names[v]
                assert not out.reshape(-1,16)[:,:15].any()
            passed.add(v)
        for v in range(23,27):
            assert lib.trial_pow(v,None,None,m,n,k,r,None,None,None,None,None,None)!=0
        print(f'PASS full transcript / final GEMM checksum / diagnostic mining rejection: {m}x{n}',flush=True)
        return passed
    finally:
        for buf in (a,b,t):buf.free()
valid=set(names)
shapes=[(256,512,11),(512,256,23),(4096,4096,42)]
if square_trial or reuse_trial: shapes += [(128,256,57),(384,768,101)]
if mapping_trial or operand_trial or prefetch_trial: shapes += [(256,768,57),(768,512,101)]
for shape in shapes:
    valid &= gate(*shape) if full_variants else (gate(*shape) | {25,26})
m=n=k=4096; r=256
rng=np.random.default_rng(100)
a=cc.DBuf(m*k);b=cc.DBuf(n*k);t=cc.DBuf((m//16)*(n//16)*64)
results={}
try:
    a.from_host(rng.integers(-128,128,(m,k),dtype=np.int8))
    b.from_host(rng.integers(-128,128,(n,k),dtype=np.int8))
    if prefetch_trial and sys.argv[1]=='ada-swizzle':
        # Bring clocks up before the short regional samples; full-grid A/B
        # remains the performance acceptance gate.
        ms=ctypes.c_float()
        cc._chk(lib.trial_bench(110,a.ptr,b.ptr,m,n,k,r,t.ptr,10000,ctypes.byref(ms)),'global warmup')
    forward=[v for v in names if v in valid] if full_variants else [11,23,25,20,24,26]
    for order in (forward,list(reversed(forward)),forward,list(reversed(forward)),forward):
        for v in order:
            info=np.empty(5,np.int32); ms=ctypes.c_float()
            cc._chk(lib.trial_info(v,info.ctypes.data_as(P)),'info')
            cc._chk(lib.trial_bench(v,a.ptr,b.ptr,m,n,k,r,t.ptr,20,ctypes.byref(ms)),'warmup')
            cc._chk(lib.trial_bench(v,a.ptr,b.ptr,m,n,k,r,t.ptr,150,ctypes.byref(ms)),'bench')
            result=results.setdefault(v,{'name':names[v],'resources':info.tolist(),'samples_ms':[]})
            result['samples_ms'].append(ms.value)
    for v,row in results.items():
        row['median_ms']=statistics.median(row['samples_ms'])
        row['equivalent_GEMM_TOPS']=2*m*n*k/(row['median_ms']/1000)/1e12
    Path('/tmp/pearl-fold-'+('ada-swizzle' if prefetch_trial and sys.argv[1]=='ada-swizzle' else 'ada-copy' if prefetch_trial and sys.argv[1]=='ada-copy' else 'ada-bpack' if prefetch_trial and sys.argv[1]=='ada-bpack' else 'ada-layout' if prefetch_trial and sys.argv[1]=='ada-layout' else 'ada-v2' if prefetch_trial and sys.argv[1]=='ada-v2' else 'ring' if prefetch_trial and sys.argv[1]=='ring' else 'rhythm' if prefetch_trial and sys.argv[1]=='rhythm' else 'ldsm-order' if prefetch_trial and sys.argv[1]=='ldsm-order' else 'ldsm' if prefetch_trial and sys.argv[1]=='ldsm' else 'prefetch2' if prefetch_trial and sys.argv[1]=='prefetch2' else 'prefetch' if prefetch_trial else 'operand' if operand_trial else 'mapping' if mapping_trial else 'reuse' if reuse_trial else 'mainloop' if mainloop_trial else 'square' if square_trial else 'l1' if cache_trial else 'occupancy' if occupancy else 'ablation')+'-results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)
finally:
    for buf in (a,b,t):buf.free()
