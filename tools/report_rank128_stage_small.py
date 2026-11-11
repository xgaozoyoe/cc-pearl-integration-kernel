"""Generate a compact reproducible report after stage and grid tests finish."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1];art=root/'artifacts'
d=json.loads((art/'rank128-stage-small-summary.json').read_text())
stage=d['stage-rotation']['events'];small=d['small-current'];grid=small['results']['summaries'];events=small['events']['summaries']
lines=['Stage rotation and smaller tiles, GPU0, 2026-10-09','',
'Production libraries unchanged; isolated experiments, no live submissions.',
'1. Read-only shared stage rotation (8192-region CUDA event time)',
'Layer        mean ms     median ms   p95 ms      registers',]
for i,n in enumerate(('fixed','pingpong','global','full')):
 r=stage['summaries'][n];reg=stage['resources'][str(i)][0]
 lines.append(f'{n:12} {r["mean_ms"]:.6f}    {r["median_ms"]:.6f}    {r["p95_ms"]:.6f}    {reg}')
a=stage['summaries'];lines += ['',f'Pingpong vs fixed time change: {(a["pingpong"]["mean_ms"]/a["fixed"]["mean_ms"]-1)*100:.3f}%.',
 f'Real global stream minus pingpong: {a["global"]["mean_ms"]-a["pingpong"]["mean_ms"]:.6f}ms.',
'All variants reserve69632B shared,384threads,1CTA/SM,12warps,zero spill.',
'Fixed repeats first64K input; pingpong loads first128K once and alternates',
'two read-only stages for64iterations. Both retain a barrier each iteration.',
'Global uses all4096K and real cp.async streaming. Full includes transcript.',
'Synthetic inputs, initialization volume, code generation and automatic',
'clocks differ: deltas are diagnostic, not isolated additive causal costs.',
'200samples/layer in8rotated blocks;100warmups/block;16launches/sample.',
'16 CPU-sink/accepted-transcript comparisons passed,2seeds and2shapes.',
'', '2. Full-transcript smaller tile test on current A-stream',
'CTA         warp tile  regs  sharedB  CTAs/SM warps/SM  kernel mean ms',]
for n,g in zip(('baseline','tile128','tile64'),d['geometry']):
 lines.append(f'{g["CTA"][0]}x{g["CTA"][1]:<7} {g["warp_tile"][0]}x{g["warp_tile"][1]:<6} {g["registers"]:<5} {g["shared_bytes"]:<8} {g["active_CTAs"]:<7} {g["resident_warps"]:<9} {events[n]["mean_ms"]:.6f}')
lines += ['', 'End-to-end offline grid (M=N131072,K4096,R128,region8192):',
'Variant     mean ms    median ms  p95 ms     grids/sec  equiv TH/s']
for n in ('baseline','tile128','tile64'):
 r=grid[n];lines.append(f'{n:12}{r["mean_ms"]:.3f}    {r["median_ms"]:.3f}    {r["p95_ms"]:.3f}    {r["grids_sec"]:.5f}    {r["equivalent_ths"]:.3f}')
for n in ('tile128','tile64'):lines.append(f'{n} throughput change: {grid[n]["throughput_change_pct"]:.3f}%.')
lines += ['',
'Kernel:240samples/variant,6rotated blocks,100warmups/block,16launches/sample.',
'Grid:60measurements/variant,6rotated blocks,2warmup+10measured grids/block.',
'Grid uses same seed sequence and actual setup/noise/Merkle/V3/GEMM/transcript/',
'PoW work. Zero target prevents proof submissions. TH/s is offline equivalent.',
'24 full transcript AND digest comparisons PASS against accepted R128,',
'2seeds x4shapes x3variants,including8192 and partial B boundary shapes.',
'Event harness also checks full8192transcripts. All candidates zero spill.',
'Current baseline SASS exactly matches prior measured baseline.',
'',
'Reuse tradeoff (computed request volume, NOT measured L2/DRAM counters):',
'Global->shared operand bytes/useful warp MMA:37.625 ->64 ->96.',
'LDSM bytes/executed warp MMA:128 ->192 ->256.',
'Baseline padding executes67633152MMA/region; smaller tiles67108864.',
'Smaller candidates thus have slightly less padding work yet remain slower.',
'',
'Clocks/power settings unchanged; clocks automatic, not locked. Per-block',
'end telemetry recorded. Smaller kernels have different observed SM clocks;',
'practical timings include this response. These tests do not isolate a',
'fixed-clock architectural effect or prove a particular hardware stall cause.',
'No new Tensor-utilization,eligible-warp or L2 counters collected.',
'',
'Conclusion: these two smaller mappings achieve16/24resident warps but lose',
'end-to-end throughput. Residency alone is not a sufficient optimization.',
'This does not rule out every smaller tile. Retain the256x192baseline.',
'Stage pointer rotation alone has little cost in the synthetic control;',
'future work should preserve reuse and examine real copy/wait/compute',
'handoff, rather than shrink tiles or rewrite swizzle addresses blindly.',
'',
'Sources/diffs:tools/build_rank128_stage_rotation.py,',
' tools/build_rank128_small_current.py,artifacts/rank128-stage-rotation-*.diff,',
' artifacts/rank128-small-current-*.diff.',
'Reproduce validation/bench with corresponding tests/bench_rank128_*.py.',
'Analyze:tools/analyze_rank128_stage_small.py; this report:',
' tools/report_rank128_stage_small.py. No commit or deployment.',]
(art/'rank128-stage-small-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
