"""Summarize isolated operand-cohort trial; no GPU operations."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1];out=root/'artifacts/operand-cohorts'
e=json.loads((out/'events.json').read_text());s=e['summaries']
lines=['Operand-cohort experiment. GPU0 only; offline CUDA events,8192x8192,K4096,R128.',
'200 samples/variant,16 launches/sample,10 rotating blocks. Clocks unlocked.',
'Each region uses the same inputs,CTA256x192,warp64x64,K64,two stages,64 static IMMA and16 LDSM.',
'Copy ownership changed without duplicating vectors:28672 logical bytes/K64/CTA. L2 traffic not measured.',
'Production unchanged. No new Nsight run or end-to-end grid timing in this experiment.',
'', 'variant mean_ms median_ms p95_ms delta_vs_native regs local_bytes active_CTAs']
for n,x in s.items():
 r=e['resources'][n]
 lines.append(f'{n} {x["mean_ms"]:.6f} {x["median_ms"]:.6f} {x["p95_ms"]:.6f} {100*(x["mean_ms"]/s["native"]["mean_ms"]-1):+.3f}% {r[0]} {r[1]} {r[3]}')
lines += ['',f'remap_named/source_control time change: {100*(s["remap_named"]["mean_ms"]/s["source_control"]["mean_ms"]-1):+.3f}%',f'cohort_ab/remap_named time change: {100*(s["cohort_ab"]["mean_ms"]/s["remap_named"]["mean_ms"]-1):+.3f}%',
'All have69632 dynamic shared bytes,384threads,1CTA/SM; candidates167regs,others168; no spills.',
'Source_control is recompiled ancestor,NOT instruction-identical to frozen native. Source_control publishes after all64IMMA; frozen native has16IMMA after publication.',
'Therefore deltas to native include lost binary scheduling improvements; they do not isolate barrier cost.',
'', 'Correctness: initial60 small/edge transcript cases PASS; all5 variants8192x8192 preflight PASS.',
'Racecheck:24 small/edge cases for cohort_ab/cohort_ba,0hazards,0errors,0warnings.',
'Full transcript+PoW digest:30 cases (5variants,3shapes,2seeds),PASS.',
'', 'Synchronization proof and limitation:',
'A-group=all3warps with same wm; B-group=all4warps with same wn.',
'Only A-group members write/read its A rows; only B-group members write/read its B rows.',
'Each warp cp.wait0 + all old-stage LDSM finishes before its first barrier.',
'A barrier96threads and B barrier128threads publish copies and protect old-stage reuse.',
'IDs1..4(A),5..7(B),0(prologue/epilogue); fixed memberships; all group members arrive once/iteration.',
'However A-then-B creates all-CTA transitive dependence: each B member arrived through a different A group.',
'B-then-A similarly spans all12warps. With no useful work between these two waits, global completion remains necessary.',
'This is not a successful elimination of global waiting. Do not promote or tune only group order.',
'Further work would need independently placed ready/reuse handoffs and useful A or B work between them; correctness and overhead are unproven.']
# Check the transitive closure directly for every warp and both orders.
for w in range(12):
 a=lambda x:{v for v in range(12) if v%4==x%4}
 b=lambda x:{v for v in range(12) if v//4==x//4}
 assert set().union(*(a(v) for v in b(w)))==set(range(12))
 assert set().union(*(b(v) for v in a(w)))==set(range(12))
(out/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
