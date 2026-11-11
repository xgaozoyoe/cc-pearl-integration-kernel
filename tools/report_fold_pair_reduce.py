"""Summarize paired-loop experiment without equating region events to full grids."""
from pathlib import Path
import json,re
p=Path(__file__).resolve().parents[1]/'artifacts/fold-pair-reduce'
e=json.loads((p/'events.json').read_text());v=json.loads((p/'validation.json').read_text())
lines=['GPU0 offline paired K64 experiment; K4096/rank128, CTA256x192,384threads,2stages.',f'Edge transcript comparisons: {len(v)} PASS. Benchmark also checks8192x8192 transcripts.', 'CUDA events:8192x8192 region,200 samples/variant,16 launches/sample,10 rotating batches.', 'Clocks/power NOT locked. No Nsight, no full-grid or pool TH/s claim.', 'variant mean_ms median_ms p95_ms vs_native_percent resources[regs,local,static_shared,active_CTAs]']
b=e['summaries']['native']['mean_ms']
for n,x in e['summaries'].items():
 lines.append(f"{n} {x['mean_ms']:.6f} {x['median_ms']:.6f} {x['p95_ms']:.6f} {100*(x['mean_ms']/b-1):+.3f} {e['resources'][n]}")
for n in ('source8','defer2_copy','defer2_mma'):
 s=(p/(n+'.sass')).read_text();lines.append(n+' static SASS counts '+str({op:len(re.findall(r'\b'+op+r'\b',s)) for op in ('IMMA','LDSM','REDUX','BRA','LDL','STL')}))
lines += ['', 'Interpretation: two K64 bodies are emitted (128staticIMMA,32LDSM), but even immediate-fold control spills.', 'Static instruction counts are not dynamic throughput. No branch-only causal claim.', 'Source recompilation also loses native fixed-SASS early publication; source8 is the matched unroll control.', 'Deferred variants retain first-pair and first/second transcript-half handling; this is not a branch-free experiment.', 'No candidate promoted. Do not spend full-grid/profiling time on these clearly slower candidates.', '', 'Next bounded design: preserve native main loop and early publication. First test moving only REDUX after boundary local XOR, retaining existing transcript addresses and store block. Avoid full reduction helper across backedge.', 'If crossing the backedge is required, separate prologue/steady-state/drain and first16/second16 folds; carry minimal reduced state, audit register reuse and all boundary writes before benchmarking.', 'Moving REDUX alone can only target a fraction of measured transcript cost (~6.73% in prior baseline); it does not eliminate local accumulator XOR dependencies.', 'Do not claim concurrent hardware execution from source or instruction interleaving alone.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
