"""Diagnostic dependency-cut results; no mining performance or causal clock confounding."""
from pathlib import Path
import json,argparse,statistics as st
import numpy as np
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1];out=root/'artifacts/native-dependency-cuts';m=json.loads((out/'manifest.json').read_text());names=list(m['variants'])
lines=['Fastest native IMMA input cuts. All but native deliberately violate Pearl arithmetic.', 'No compiler:all instruction PCs/counts and64staticIMMA/16LDSM/copy/transcript/barrier sites retained.', 'a_zero:A=RZ. b_zero:B=RZ. ab_zero:A=B=RZ. bc_zero:B=C=RZ.', 'C denotes old INT32 accumulator input; destination register remains unchanged.', 'b_release/bc_release move3B-result waits after removing B-input consumption;', 'waits remain before original B destination registers are overwritten. All other waits/fixed stalls preserved.', 'First-half three wait moves:11e0->1220,1290->12d0,1350->1410.', 'Input substitution is literalzero, not prefetched real B. No compiler DCE or new initialization.', 'C cut does not remove destination hazards, fixed scheduling delays or all dependencies.', 'No output hashing, share submission or full-grid mining-rate claims for diagnostic variants.', '']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());lines+=['UNLOCKED-CLOCK region events8192x8192x4096:200samples/variant,16launches/sample,10rotated batches.', 'name mean_ms median_ms p95_ms']
 for name in names:
  x=e['summaries'][name];lines.append(f"{name} {x['mean_ms']:.6f} {x['median_ms']:.6f} {x['p95_ms']:.6f}")
 for left,right in [('native','a_zero'),('native','b_zero'),('b_zero','b_release'),('b_zero','bc_zero'),('b_release','bc_release')]:
  x=e['summaries'];lines.append(f'{left}->{right}:time change {100*(x[right]["mean_ms"]/x[left]["mean_ms"]-1):+.3f}%')
 lines+=['IMPORTANT:zero operands change power/clock behavior. Unlocked wall-time differences are NOT isolated dependency costs. Require same-clock,cycle-normalized counters before deciding bottleneck.', 'Resources '+json.dumps(e['resources'])]
if a.capture:
 cap=a.capture;issue=json.loads((cap/'imma-issue.json').read_text());summary=json.loads((cap/'summary.json').read_text());lines+=['','FIXED-BASE-CLOCK CAPTURE']
 for i,name in enumerate(names):
  lines.append(name+' '+json.dumps(issue[str(i)]['means']));lines.append(json.dumps({k:v['mean'] for k,v in summary[str(i)]['metrics'].items()}))
  assert all(r['count_matches_sass'] for r in issue[str(i)]['launches']),name
 for left,right in [('native','a_zero'),('native','b_zero'),('b_zero','b_release'),('b_zero','bc_zero'),('b_release','bc_release')]:
  l=issue[str(names.index(left))]['means']['imma_per_SM_elapsed_cycle'];r=issue[str(names.index(right))]['means']['imma_per_SM_elapsed_cycle'];lines.append(f'{left}->{right}:IMMA/SM-cycle change {100*(r/l-1):+.3f}%')
 (cap/'dependency-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
