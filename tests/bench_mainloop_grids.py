"""GPU-0 normal full-grid A/B comparison, with deterministic matched inputs."""
import os,sys,json,statistics,ctypes
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ.setdefault('PEARL_TRIAL_LIBRARY','/tmp/libpearl-pure-gemm.so')
candidate=int(sys.argv[1]) if len(sys.argv)>1 else 53
baseline=int(sys.argv[2]) if len(sys.argv)>2 else 20
assert baseline in (20,61,67,88,89,104,109,110) and baseline != candidate
assert candidate in range(53,122)
names=[str(i) for i in range(122)]
names[0]='production';names[20]='current best';names[53]='B LDSM x4'
names[54]='K128 s2 streaming';names[55]='K128 s2 full load'
names[56]='K32 lookahead joint';names[57]='K32 lookahead staggered'
for v,g in ((58,1),(59,4),(60,8),(61,16)): names[v]=f'group M{g}'
for v,g in ((62,2),(63,12),(64,24)): names[v]=f'group M{g}'
for v,g in ((65,2),(66,4),(67,8)): names[v]=f'group N{g}'
names[68]='256x128 warp4x2';names[69]='256x128 warp2x4';names[70]='256x128 warp8x1'
names[71]='128x128 warp2x2';names[72]='128x128 warp1x4';names[73]='128x128 warp4x1'
names[74]='128x320 register transcript';names[75]='128x320 direct transcript'
names[76]='128x320 shared transcript s3';names[77]='128x320 shared transcript s4'
names[78]='wide stream B shared transcript';names[79]='wide stream A/B shared transcript'
names[80]='wide stream B register transcript';names[81]='baseline stream B'
names[82]='256x192 stream B s3';names[83]='256x192 stream A/B s3'
names[84]='256x192 register transcript';names[85]='256x192 stream B s4'
names[86]='256x192 K64 stream B s3';names[87]='256x192 K64 stream A/B s3';names[88]='256x192 K64 stream B s2'
names[89]='B prefetch before group';names[90]='B prefetch after first half';names[91]='B prefetch midway first half'
names[92]='B prefetch s3';names[93]='dead-row A prefetch s2';names[94]='dead-row A prefetch s3'
names[97]='rotating MMA rows';names[98]='interleaved MMA halves'
names[99]='K64 rolling B / bulk A';names[100]='K64 rolling B / dead-row A'
names[101]='three-slot B ring';names[102]='two-slot B ring'
names[103]='hand Ada v2 lead1';names[104]='hand Ada v2 lead2'
names[105]='hand Ada v2 streamed B';names[106]='hand Ada v2 K32 prefetch'
names[107]='K64 planar K32';names[108]='K96 planar K32'
names[109]='raw address regrouping'
names[110]='MMA-packed B x4'
names[111]='unrolled async copies'
names[112]='fixed async copy rounds'
names[113]='packed B three stages'
names[114]='fixed copies three stages'
names[115]='warp 64x96 operand reuse'
names[116]='no swizzle'
names[117]='xor3 shift3'
names[118]='xor2 shift2'
names[119]='xor3 shift2'
names[120]='ordered async producer lanes'
names[121]='ordered producer direct shared address'
os.environ['PEARL_TRIAL_NAMES']=json.dumps(names)
import bench_cute_grid as grid
candidate_library=grid.lib
baseline_library=candidate_library
if os.environ.get('PEARL_BASELINE_LIBRARY'):
    baseline_library=ctypes.CDLL(os.environ['PEARL_BASELINE_LIBRARY'])
    for name in ('trial_pow','trial_info'):
        getattr(baseline_library,name).argtypes=getattr(candidate_library,name).argtypes
original_setup=grid.cc.setup_job
counts={0:0,baseline:0,candidate:0}
class Scheduler(grid.Scheduler):
    def maybe_switch(self):
        self.grids+=1
        return self.grids>=12
grid.Scheduler=Scheduler
results=[]
try:
    for v in (0,baseline,candidate,candidate,baseline,0):
        grid.lib=baseline_library if v==baseline else candidate_library
        def setup(*args):
            seed=0x504541524C000000+counts[v];counts[v]+=1
            return original_setup(*args[:-1],seed)
        grid.cc.setup_job=setup
        results.append(grid.run(v))
finally:
    grid.cc.setup_job=original_setup
    grid.lib=candidate_library
summary={}
for v in (0,baseline,candidate):
    ms=[t*1000 for r in results if r['variant']==v for t in r['times'][2:]]
    assert len(ms)==20
    med=statistics.median(ms)
    summary[v]={'median_ms':med,'mean_ms':statistics.mean(ms),
        'p95_ms':sorted(ms)[18]*.95+sorted(ms)[19]*.05,
        'grids_per_second':1000/med,'effective_THS':70.368744177664/(med/1000)}
out={'baseline':baseline,'baseline_library':baseline_library._name,'candidate_library':candidate_library._name,
     'warmup_per_block':2,'stable_grids_per_variant':20,'results':results,'summary':summary}
Path(os.environ.get('PEARL_GRID_RESULT_PATH',f'/tmp/pearl-mainloop-grid-v{candidate}-results.json')).write_text(json.dumps(out,indent=2))
print(json.dumps(summary,indent=2),flush=True)
