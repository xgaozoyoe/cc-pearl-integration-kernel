"""Report failed cross-K copy offset experiment without promoting it."""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parents[1];art=root/'artifacts';x=json.loads((art/'handoff-cursor-events.json').read_text());names=['baseline','control','hoist','induction'];s={}
lines=['GPU0 cross-K copy-offset experiment; production and lean checkpoint unchanged.',
'Fixed tile256x192,K64,2stages,384threads,12warps,K4096,R128,64IMMA+16LDSM.',
'All4 variants168registers/thread,0local/spill,69632Bshared,1CTA/SM.',
'Baseline and lean-control instruction text/PCs exactly match previous binaries.',
'240 CUDA event samples/variant,16launches/sample,6rotated blocks after warmup.',
'Clock/power configuration unchanged;DVFS not locked. Times exclude full-grid setup and PoW.','']
for v,name in enumerate(names):
 a=np.array([t for r in x['measurements'] if r['variant']==v for t in r['samples_ms']]);s[name]=dict(samples=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)))
 lines.append(f'{name:10} mean={a.mean():.6f} median={np.median(a):.6f} p95={np.percentile(a,95):.6f} ms')
for v,name in [(2,'hoist'),(3,'induction')]:
 pair=[]
 for repeat in range(6):
  b=next(r['mean_ms'] for r in x['measurements'] if r['variant']==1 and r['repeat']==repeat)
  c=next(r['mean_ms'] for r in x['measurements'] if r['variant']==v and r['repeat']==repeat);pair.append(100*(c/b-1))
 lines.append(f"{name} slowdown vs lean checkpoint={100*(s[name]['mean_ms']/s['control']['mean_ms']-1):.3f}%;paired block slowdowns%={pair}")
lines+=['', 'Source intent:',
'hoist caches each thread row origins, retains origin+(kt+1)*64 calculation.',
'induction retains copy cursors across K iterations and adds64bytes each iteration.',
'No change to copied elements,partial B zero-fill,shared stage size or handoff algorithm.',
'', 'SASS findings:',
'Static loop-body instructions control490,hoist479,induction476;these include alternative fold branches and are NOT dynamic instruction counts.',
'Control future-copy wait after48IMMA;both candidates after56IMMA. Arrival remains after64IMMA and before transcript.',
'All main-loop copy commits still precede the first IMMA. No evidence here that the wait position change alone causes the regression.',
'Compiler scheduling/register assignment changed along with the source simplification; unchanged register count does not establish unchanged dependency behavior.',
'', 'Validation:',
'32 full transcript+PoW comparisons passed,including8192x8192.',
'ComputeSanitizer2025.2.1:racecheck24 small/edge cases0hazards;full-size memcheck0errors;synccheck24small/edge cases0errors.',
'', 'Decision:',
'Reject both candidates for promotion. Preserve verified lean checkpoint. No full-grid test or end-to-end gain claim for these slower candidates.',
'No V3,target,proof,production backend or live mining changes.',
'Optional next diagnostic:profile baseline/control/hoist/induction to distinguish dynamic instruction counts,IMMA/SMcycle and PC-stall changes.',
'Capture: sudo bash tools/profile_handoff_cursor.sh 2270585',
'Do not attribute regression to dependency stalls without those counters.']
(art/'handoff-cursor-report.txt').write_text('\n'.join(lines)+'\n');(art/'handoff-cursor-summary.json').write_text(json.dumps(s,indent=2)+'\n');print('\n'.join(lines[:14]))
