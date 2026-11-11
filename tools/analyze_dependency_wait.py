"""Summarize a completed, hash-verified GPU0 wait-relocation capture."""
from pathlib import Path
import argparse,json,collections,hashlib,shutil
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);args=p.parse_args();src=args.capture.resolve()
for line in (src/'inputs.sha256').read_text().splitlines():
 digest,name=line.split(maxsplit=1);path=root/name.strip();assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
out=root/'artifacts/dependency-wait/profile'
if out.exists():
 assert (out/'inputs.sha256').read_bytes()==(src/'inputs.sha256').read_bytes()
else:shutil.copytree(src,out)
s=json.loads((src/'summary.json').read_text());issues=json.loads((src/'imma-issue.json').read_text());pcs=json.loads((src/'pc-stalls.json').read_text())
names=['baseline','ab_zero','ab_wait2','ab_wait3'];reference=issues['1']['means']['imma_per_SM_elapsed_cycle'];result={}
for i,name in enumerate(names):
 issue=issues[str(i)];assert len(issue['launches'])==3 and all(x['count_matches_sass'] for x in issue['launches'])
 means=issue['means'];metrics=s[str(i)]['metrics'];rep=next(x for x in pcs if Path(x['file']).name==f'v{i}.sass.csv')
 r=dict(kernel_ms=means['kernel_ms'],imma_per_SM_cycle=means['imma_per_SM_elapsed_cycle'],cycle_throughput_gain_vs_ab_zero_pct=100*(means['imma_per_SM_elapsed_cycle']/reference-1),l2_bytes_per_useful_mma=s[str(i)]['l2_bytes_per_useful_mma'])
 r.update({k:metrics[k]['mean'] for k in ['tensor_pct','short_scoreboard_pct','math_throttle_pct','barrier_pct','eligible_warps_per_scheduler','issued_warps_per_cycle']})
 r['pc_samples']={}
 for reason in ['stall_short_sb','stall_math']:
  c=collections.Counter()
  for x in rep['pcs']:c[x['pc_offset']]+=x['samples'].get(reason,0)
  r['pc_samples'][reason]={'selected':{pc:c[pc] for pc in ['0x11a0','0x11e0','0x1250','0x1290','0x1310','0x1420']},'top':c.most_common(6)}
 result[name]=r
(out/'interpretation.json').write_text(json.dumps(result,indent=2)+'\n')
lines=[f'Capture: {src}', 'All captured input hashes match; 3 profiled launches/variant, each 67,633,152 IMMA.', '', 'Variant     kernel_ms  Tensor%  shortSB%  mathThrottle%  IMMA/SMcycle  gain_vs_ABzero%']
for name,r in result.items():lines.append(f"{name:11} {r['kernel_ms']:.6f}   {r['tensor_pct']:.3f}   {r['short_scoreboard_pct']:.3f}     {r['math_throttle_pct']:.3f}       {r['imma_per_SM_cycle']:.9f}   {r['cycle_throughput_gain_vs_ab_zero_pct']:+.3f}")
lines+=['', 'AB-zero -> 3 waits relocated: short scoreboard decreases ~40%; per-cycle IMMA throughput increases only ~0.27%.', 'Selected short-scoreboard PC samples, summed across 3 launches (not time fractions):', 'PC          AB-zero    wait3']
for pc,n in result['ab_zero']['pc_samples']['stall_short_sb']['selected'].items():lines.append(f"{pc:10} {n:8.0f} {result['ab_wait3']['pc_samples']['stall_short_sb']['selected'][pc]:8.0f}")
lines+=['', 'Interpretation: targeted waits were successfully hidden, with limited residual waiting at new locations.', 'A large reduction in this warp stall metric does not imply a similar wall-time reduction.', 'Higher math throttle is consistent with ready IMMA work encountering issue/pipeline backpressure; it does not prove its root cause.', 'Eligible warps/scheduler remains rounded to 0.33; issued warps/cycle remains rounded to 0.22.', 'All L2 bytes/useful MMA remain ~37.759. No work-count or operand-reuse change explains the result.', 'Only three B windows were tested. Other LDSM waits, fixed scheduling stalls and IMMA issue cadence remain unresolved.', 'Next investigative priority: map remaining wait sites (10e0,1500) and hot IMMA sequences (1210,1500,1580), separating encoded stalls from operand waits and math-pipe backpressure.', 'Do not remove fixed stalls indiscriminately: that can violate C accumulator dependencies.', 'No production change justified by this result; all AB-zero variants intentionally compute invalid Pearl transcripts.']
(root/'artifacts/dependency-wait/profile-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
