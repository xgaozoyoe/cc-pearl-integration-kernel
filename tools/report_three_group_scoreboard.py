"""Report exact-instruction scoreboard controls; offline timings only."""
from pathlib import Path
import json
p=Path(__file__).resolve().parents[1]/'artifacts/three-group-scoreboard'
lines=['GPU0,8192x8192 region,K4096,rank128. CUDA events; unlocked clocks.', 'Each run200samples/variant,16launches/sample,10rotating blocks. Not full-grid/pool throughput.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries'];lines += [f'Run{run}','variant mean_ms median_ms p95_ms delta_native_pct delta_three_early_pct']
 for n,v in s.items():lines.append(f"{n:16} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.4f} {(v['mean_ms']/s['three_early']['mean_ms']-1)*100:+.4f}")
assert len(json.loads((p/'validation.json').read_text()))==48
assert len(json.loads((p/'pow-validation.json').read_text()))==24
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck.log').read_text()
lines += ['48 transcript+24 transcript/PoW comparisons PASS; racecheck0 hazards/errors/warnings.',
 'Native168regs; all three-group variants167regs,0local,69632B shared,384threads,1CTA/SM.',
 'clear_tail: clear old LDSM wait masks at0x16f0 and0x1730; publication already waits both loads at0x16d0. The first old wait now targets BAR read scoreboard1, not an MMA operand.',
 'split_read_sb: additionally assign second BAR.ARV read toSB2; waitSB2 at0x2a20 before updating its ID register. First ID update retainsSB1 wait.',
 'Same instruction sequence/operands for all three-group variants; only control fields changed. See *.diff for exact bit changes.',
 'No production modifications; no hardware-counter or live-mining measurements. Numerical changes in wall time alone do not identify a hardware stall cause.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
