"""Summarize events and optional Nsight wide CSV capture; never label probes mining gains."""
from pathlib import Path
import argparse,json,csv,statistics
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'artifacts/handoff-abcd'
names=list(json.loads((out/'manifest.json').read_text())['variants'])
lines=['A/B/C0/C/D: offline 8192x8192 K4096 region; NOT full-grid mining throughput.', 'CAUTION: unlocked event runs have different boost clocks; time differences are not isolated dependency gains. C/C0 sampled near2715MHz while B varied2460..2670MHz. Use fixed-clock Nsight capture and per-cycle normalization.']
lines.insert(0, 'SCOPE CORRECTION: these historical C/D variants are LOCAL probes, not a complete initialization ABCD experiment. C changes B operand sourcing and three waits; D removes three copy-guard operations. No conclusion about the complete initialization dependency chain is established. See handoff-abcd-scope-audit.txt.')
for path in (sorted(out.glob('events-run*.json')) or [out/'events.json']):
 s=json.loads(path.read_text())['summaries'];base=s['B']['mean_ms'];lines+=['',path.name,'Variant  median_ms  mean_ms  p95_ms  time_vs_B']
 for n in names:
  r=s[n];lines.append(f"{n:7s} {r['median_ms']:.6f} {r['mean_ms']:.6f} {r['p95_ms']:.6f} {(r['mean_ms']/base-1)*100:+.3f}%")
metrics=['sm__cycles_elapsed.avg.per_second','gpu__time_duration.sum','smsp__inst_executed_pipe_alu.sum','smsp__inst_executed_pipe_tensor.sum','smsp__inst_executed_pipe_tensor_v2.sum','smsp__warps_eligible.avg.per_cycle_active','smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct','smsp__warp_issue_stalled_barrier_per_warp_active.pct']
if a.capture:
 result={}
 for v,n in enumerate(names):
  rows=list(csv.reader((a.capture/f'v{v}.raw.csv').open()));i=next(i for i,r in enumerate(rows) if 'ID' in r and 'Kernel Name' in r);h=rows[i]
  # Nsight inserts a units row (e.g. "hz") after the header. Only
  # launch records have a numeric ID; do not silently ignore bad metrics.
  ds=[dict(zip(h,r)) for r in rows[i+1:]
      if len(r)==len(h) and r[h.index('ID')].strip().isdigit()]
  if not ds:raise ValueError(f'No launch records in {a.capture / f"v{v}.raw.csv"}')
  result[n]={}
  for m in metrics:
   vals=[float(d[m].replace(',','')) for d in ds if m in d and d[m] not in ('','n/a','N/A')]
   result[n][m]=statistics.mean(vals) if vals else None
 (out/'counters.json').write_text(json.dumps(result,indent=2)+'\n');lines+=['',json.dumps(result,indent=2)]
lines+=['','C/C0 are invalid Pearl arithmetic. C vs C0 isolates only three deferred B waits.', 'D retains NOP slots; negative results cannot exclude a benefit from rescheduling.', 'All variants:168 regs/thread,69632B dynamic shared,0 local bytes,1CTA/SM,12warps/SM.']
text='\n'.join(lines)+'\n';(out/'report.txt').write_text(text);print(text)
