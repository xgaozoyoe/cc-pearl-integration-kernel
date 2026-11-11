"""Frozen SASS local fold eligibility-window experiment."""
from pathlib import Path
import json
p=Path(__file__).resolve().parents[1]/'artifacts/fold-ready-window'
lines=['GPU0 only.8192x8192 region,K4096,rank128. CUDA event kernel timings,not fullgrid/pool.',
'Two independent runs,200samples/variant/run,16launches/sample,10rotating blocks;unlocked clocks.',
'First block3localXOR moved before/after DEPBAR16c0. Full REDUX and stores stay late.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries']
 lines.append(f'Run{run}: name mean_ms median_ms p95_ms delta_native_pct delta_after_wait_pct')
 for n,v in s.items():lines.append(f"{n:12} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.4f} {(v['mean_ms']/s['after_wait']['mean_ms']-1)*100:+.4f}")
assert len(json.loads((p/'validation.json').read_text()))==36
assert len(json.loads((p/'pow-validation.json').read_text()))==18
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck.log').read_text()
lines+=['36transcript+18transcript/PoW comparisons PASS;racecheck0 hazards/errors/warnings.',
'All168regs,0local,69632B shared,384threads,1CTA/SM. Same64static MMA and16LDSM.',
'Both controls add same2branches/iteration,3NOPs/fold and perform3XOR also on nofold iterations. This is diagnostic overhead,not a production optimization.',
'R165 is existing dead temporary;3local XOR use6-cycle static spacing;remaining XOR18c0 readsR165.',
'Initial reordering failed because extracted XORs lost intervening instruction latency;rename-only control passed. Corrected contiguous chain passed. Pre-chain padding alone did not fix it.',
'No barriers removed or added. Copy wait and peer publication retained;all proof/transcript semantics preserved.',
'This tests a small part of ready-work availability,not complete early transcript or independent transcript warps.',
'No hardware counters measured yet. Eligible warps/cycle or actual overlap cannot be inferred from event timing.',
'Counters require sudo bash tools/profile_fold_ready_window.sh (GPU0 only).',
'No production changes,no live mining. See dependency-map.txt and candidate.diff for audit.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
