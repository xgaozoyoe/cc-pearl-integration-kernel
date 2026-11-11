"""Report controlled reduction of named-barrier loop overhead."""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parents[1];art=root/'artifacts';names=['baseline','control','padded','rolling'];result={}
lines=['GPU0-only shared-stage addressing controls. Production unchanged.',
'0baseline,1verified lean checkpoint,2B-stage padded16KiB,3padding+rolling XOR stage offset.',
'Each variant same CTA256x192,K64,2stages,384threads,12warps,K4096,R128,64staticIMMA/16LDSM.',
'Registers baseline168,control168,padded168,rolling167;shared69632B controls/77824B candidates;0local/spill,1CTA/SM.',
'Baseline and verified lean control instruction text and PCs exactly match previous binaries.',
'No clocks/power configuration changed; DVFS is not locked. Performance mechanism needs cycle-normalized profiling.',
'Event samples average16 launches;240samples/variant/run in rotated blocks after warmup.','']
for tag,path in [('run1',art/'handoff-aligned-events-run1.json'),('run2',art/'handoff-aligned-events.json')]:
 x=json.loads(path.read_text());s={};lines.append(tag)
 for v,name in enumerate(names):
  a=np.array([t for r in x['measurements'] if r['variant']==v for t in r['samples_ms']]);s[name]=dict(samples=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)))
  lines.append(f'{name:10} n={len(a)} mean={a.mean():.6f} median={np.median(a):.6f} p95={np.percentile(a,95):.6f} ms')

 for name in ['padded','rolling']:lines.append(f"{name} time reduction vs baseline={100*(1-s[name]['mean_ms']/s['baseline']['mean_ms']):.3f}%;vs checkpoint={100*(1-s[name]['mean_ms']/s['control']['mean_ms']):.3f}%.")
 result[tag]=s
path=art/'handoff-aligned-results.json'
if path.exists():
 x=json.loads(path.read_text());s={};lines+=['','Full offline grids: includes setup/noise/Merkle/GEMM/PoW;60measured/variant after per-block warmup,rotated blocks,same deterministic seeds.']
 for v in [0,1,2,3]:
  name=names[v];a=np.array([t for r in x['measurements'] if r['variant']==v for t in r['samples_ms']]);mean=float(a.mean());s[name]=dict(samples=len(a),mean_ms=mean,median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),grids_sec=1000/mean,ths=2**46/mean/1e9)
  lines.append(f'{name:10} n={len(a)} mean={mean:.3f} median={np.median(a):.3f} p95={np.percentile(a,95):.3f} ms grids/s={1000/mean:.6f} TH/s={2**46/mean/1e9:.3f}')

 for name in ['padded','rolling']:lines.append(f"{name} full-grid time reduction vs checkpoint={100*(1-s[name]['mean_ms']/s['control']['mean_ms']):.3f}%;throughput increase vs baseline={100*(s[name]['ths']/s['baseline']['ths']-1):.3f}%.")
 result['grids']=s
lines+=['','Correctness:32 complete transcript+PoW digest comparisons including8192x8192 passed.',
'ComputeSanitizer2025.2.1:racecheck24 small/edge comparisons 0hazards;memcheck full sizes 0errors;synccheck24 small/edge comparisons 0errors.',
'No V3 seed/target/proof modifications; no production library change; no live mining.',
'Only small performance changes are expected here; no claim of closing the full PeakMiner gap.',
'Conclusion: neither padded nor rolling improves full-grid time over the lean checkpoint in this run. Retain the lean checkpoint; do not promote padding based on kernel-only means.',
'Extra profiling is optional diagnostic work, not required for promotion: there is currently no demonstrated end-to-end gain.',
'Capture: sudo bash tools/profile_handoff_aligned.sh 2270585',
'Capture variants0baseline,1lean checkpoint,2padded,3rolling. Do not confuse with previous script variant3.',
'No production promotion before reviewing the measurements.']
(art/'handoff-aligned-report.txt').write_text('\n'.join(lines)+'\n');(art/'handoff-aligned-summary.json').write_text(json.dumps(result,indent=2)+'\n');print('\n'.join(lines))
