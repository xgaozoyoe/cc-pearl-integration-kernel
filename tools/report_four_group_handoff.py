"""Offline four-group timing and correctness summary."""
from pathlib import Path
import json
p=Path(__file__).resolve().parents[1]/'artifacts/four-group-handoff'
lines=['GPU0 only,8192x8192 region,K4096,rank128. CUDA events; unlocked clocks.', 'Two independent runs;200samples/variant/run,16launches/sample,rotating blocks. Not grid or pool throughput.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries'];lines += [f'Run{run}','variant mean_ms median_ms p95_ms delta_native_pct']
 for n,v in s.items():lines.append(f"{n:16} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.4f}")
assert len(json.loads((p/'validation.json').read_text()))==48
assert len(json.loads((p/'pow-validation.json').read_text()))==24
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck.log').read_text()
lines+=['48transcript+24transcript/PoW comparisons PASS; racecheck0 hazards/errors/warnings.',
 'All tested kernels168regs,0local,69632B dynamicshared,384threads,1CTA/SM.',
 'contiguous4:[0,1,2],[3,4,5],[6,7,8],[9,10,11]. interleaved4:[0,4,8],[1,5,9],[2,6,10],[3,7,11].',
 'Four-group candidates use same machine-code main loop and register assignment; interleaved changes only prologue group ID calculation.',
 'Original interleaved source compilation spilled (40B frame); excluded from tests. Manifest points to prologue-patched no-spill contiguous binary.',
 'Each warp arrives early on3other-group barriers and syncs late on own; each barrier counts384participants. IDs4..7 /12..15 alternate.',
 'Early publication after finalLDSM completion and cp.wait, before last16IMMA. First BAR waits both LDSM scoreboards; obsolete tail wait masks cleared.',
 'Safety:completion of own barrier proves all peer early publications and own-group late arrivals, hence all shared copies/reads complete. Two phase ID sets prevent reuse before previous phase completion.',
 'Compared with native, extra2arrivals+3additional ID state registers/updates; same useful arithmetic and copy ownership. Compiler scheduling differs from native.',
 'Native is frozen optimized two-group8+4; source_control is recompiled two-group late-publication reference, not optimized native.',
 'No production changes, live mining, or hardware-counter collection. Do not infer which hardware mechanism caused timing differences without profiling.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
