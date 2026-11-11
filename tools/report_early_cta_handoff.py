"""Early full-CTA handoff, correct candidate; production untouched."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1];out=root/'artifacts/early-cta-handoff'
lines=['Frozen native two-site SASS experiment, GPU0 only.',
'early_cta:16d0 BAR.ARV -> BAR.SYNC0;2a00 BAR.SYNC ->NOP.',
'late_cta:16d0 BAR.ARV ->NOP;2a00 BAR.SYNC ->BAR.SYNC0 (negative control).',
'Original instruction positions,cp.wait,LDSM completion masks,64IMMA,16LDSM and transcript unchanged.',
'All168regs,69632dynamic shared bytes,384threads,1CTA/SM,0local bytes.',
'Early handoff occurs after this warp future copies and old-stage reads complete, before16tailIMMA.',
'All warps contribute early; next iteration no longer depends on own-group peers reaching the late sync.',
'Hardware deferred blocking may allow independent instructions before actual wait; do not equate source issue PC with stall PC.',
'Transcript accesses are warp-owned before final CTA barrier; initial/final barriers remain.',
'Original dead own/peer ID arithmetic remains to keep this a controlled test.',
'12early_cta small/edge transcript cases:racecheck0hazards,0errors,0warnings.',
'']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());s=e['summaries'];b=s['native']['mean_ms']
 lines.append('Unlocked region8192x8192,K4096,R128;200samples/variant,16launches/sample,10rotating blocks.')
 for n,v in s.items():lines.append(n+' '+json.dumps(v)+f' time_change_vs_native_pct={100*(v["mean_ms"]/b-1):+.6f}')
if (out/'grids.json').exists():
 import numpy as np
 e=json.loads((out/'grids.json').read_text());lines.append('Offline full-grid wall time;2^46equivalent hashes/grid,not pool rate.')
 for n in ['native','early_cta']:
  vals=[v for r in e['measurements'] if r['name']==n for v in r['samples_ms']]
  if vals:lines.append(n+' '+json.dumps(dict(n=len(vals),mean_ms=float(np.mean(vals)),median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),equivalent_ths=2**46/(np.mean(vals)/1000)/1e12)))
if (out/'pow-validation.json').exists():
 rows=json.loads((out/'pow-validation.json').read_text());assert all(r['transcript_pass'] and r['digest_pass'] for r in rows);lines.append(f'Full transcript+PoW digest:{len(rows)}cases PASS.')
(out/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
