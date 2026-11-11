"""Summarize source-only handoff experiments; no production promotion."""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parents[1];art=root/'artifacts'
lines=['GPU0 source-only stage handoff experiments; production path unchanged.','',
'Fixed CTA256x192,K64,2stages,384threads,12warps,K4096,R128;64staticIMMA/16LDSM.',
'Event sample = mean of16 launches,240 samples/variant/run in rotated blocks.',
'GPU clocks/power settings untouched; DVFS not locked. Differences are descriptive, not cycle-normalized.',
'First experiment: parity spin vs token vs one-warp polling+CTA broadcast.']
def summary(path,names):
 x=json.loads(path.read_text());s={}
 for v,name in enumerate(names):
  a=np.array([t for r in x['measurements'] if r['variant']==v for t in r['samples_ms']])
  if not len(a):continue
  s[name]=dict(samples=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)))
  lines.append(f'{name:12} n={len(a):3} mean={a.mean():.6f} median={np.median(a):.6f} p95={np.percentile(a,95):.6f} ms')
 return s
wait=summary(art/'handoff-wait-modes-events.json',['baseline','spin','token','delegate'])
lines+=['Token and delegate do not improve on baseline. All4 variants168regs,0local,1CTA/SM.','',
'Second experiment: two symmetric groups; named barriers replace mbarrier polling.']
results={}
for tag,path in [('run1',art/'handoff-named-events-run1.json'),('run2',art/'handoff-named-events.json')]:
 lines.append(tag);s=summary(path,['baseline','spin','late','overlap']);results[tag]=s
 lines.append(f"Overlap time reduction vs late={100*(1-s['overlap']['mean_ms']/s['late']['mean_ms']):.3f}%; vs baseline={100*(1-s['overlap']['mean_ms']/s['baseline']['mean_ms']):.3f}%.")
lines+=['Named variants167regs,0spill,shared69632B,1CTA/SM. Baseline168regs,shared69632B; spin168regs,69640B.',
'No polling LDS/R2UR loop, no mbarrier ATOMS in named variants. One BAR.ARV and one receiving BAR per iteration.',
'Four alternating IDs prevent double arrival to an unreset peer phase. All12 warps compute, none dedicated to movement.',
'Overlap moves arrival before transcript; local cp.async.wait_group0 remains BEFORE arrival.',
'SASS: overlap wait1690,arrival1800,firstREDUX1970; late firstREDUX1910,wait2a00,arrival2a70.',
'Thus this measures overlap of inter-warp handoff and transcript, not elimination of copy completion waiting.','']
grid=art/'handoff-named-results.json'
if grid.exists():
 lines.append('Full offline grids (includes setup/noise/Merkle/GEMM/PoW),same deterministic seed sequence;60 measured/variant after per-block warmup:')
 g=json.loads(grid.read_text());gs={}
 for v,name in [(0,'baseline'),(3,'overlap')]:
  a=np.array([t for r in g['measurements'] if r['variant']==v for t in r['samples_ms']]);mean=float(a.mean());gs[name]=dict(mean_ms=mean,median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),samples=len(a),grids_sec=1000/mean,ths=2**46/mean/1e9)
  lines.append(f'{name:12} n={len(a)} mean={mean:.3f} median={np.median(a):.3f} p95={np.percentile(a,95):.3f} ms grids/s={1000/mean:.6f} TH/s={2**46/mean/1e9:.3f}')
 lines.append(f"Full-grid time reduction={100*(1-gs['overlap']['mean_ms']/gs['baseline']['mean_ms']):.3f}%.");results['grids']=gs
lines+=['','Correctness: each suite32 complete transcript+PoW comparisons,including8192x8192,passed.',
'Both suites:ComputeSanitizer2025.2.1 racecheck24 small/edge cases:0hazards;full-size memcheck:0errors.',
'Named-barrier synccheck2025.2.1:0errors on24 small/edge comparisons.',
'Earlier sleep/NOP experiment remains quarantined: its racecheck issue is unresolved; not used here.',
'No live mining; no production/V3/target/proof changes; no net production speedup claimed.',
'Tensor density has NOT been measured for these variants yet. Required next evidence: same-metric IMMA/SMcycle,Tensor utilization,copy-wait and barrier PC stalls.',
'Next capture: sudo bash tools/profile_handoff_named.sh 2270585',
'References: named-barrier semantics and repeated-arrival restriction:',
'https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html']
(art/'handoff-named-report.txt').write_text('\n'.join(lines)+'\n');(art/'handoff-named-summary.json').write_text(json.dumps(results,indent=2)+'\n');print('\n'.join(lines))
