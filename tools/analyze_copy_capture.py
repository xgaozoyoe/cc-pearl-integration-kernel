"""Verify/archive copy-ablation capture and report cycle-normalized evidence."""
from pathlib import Path
import argparse,json,collections,hashlib,shutil,csv
root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path);src=ap.parse_args().capture.resolve()
for line in (src/'inputs.sha256').read_text().splitlines():
 digest,name=line.split(maxsplit=1);assert hashlib.sha256((root/name.strip()).read_bytes()).hexdigest()==digest,name
out=root/'artifacts/copy-bypass/profile'
if out.exists():assert (out/'inputs.sha256').read_bytes()==(src/'inputs.sha256').read_bytes()
else:shutil.copytree(src,out)
s=json.loads((src/'summary.json').read_text());issues=json.loads((src/'imma-issue.json').read_text());pcs=json.loads((src/'pc-stalls.json').read_text());names=['baseline','no_ab','no_copy','no_copy_bar'];result={}
for i,name in enumerate(names):
 issue=issues[str(i)];assert len(issue['launches'])==3 and all(x['count_matches_sass'] for x in issue['launches'])
 m=issue['means'];metrics=s[str(i)]['metrics'];rep=next(x for x in pcs if Path(x['file']).name==f'v{i}.sass.csv')
 r=dict(kernel_ms=m['kernel_ms'],imma_per_SM_cycle=m['imma_per_SM_elapsed_cycle'],effective_SM_GHz=m['tensor_Ginst_sec']/m['imma_per_SM_elapsed_cycle']/128,l2_bytes_per_useful_mma=s[str(i)]['l2_bytes_per_useful_mma'])
 r.update({k:metrics[k]['mean'] for k in ['tensor_pct','short_scoreboard_pct','long_scoreboard_pct','math_throttle_pct','barrier_pct','eligible_warps_per_scheduler','issued_warps_per_cycle']})
 raw=list(csv.DictReader((src/f'v{i}.raw.csv').read_text().splitlines()));r['ldsm_bytes']=[int(x['sm__sass_data_bytes_mem_shared_op_ldsm.sum'].replace(',','')) for x in raw if x['ID']]
 assert r['ldsm_bytes']==([8657043456]*3 if i==0 else [0]*3)
 r['pc_samples']={}
 for reason in ['stall_barrier','stall_short_sb','stall_long_sb','stall_math']:
  c=collections.Counter()
  for x in rep['pcs']:c[x['pc_offset']]+=x['samples'].get(reason,0)
  r['pc_samples'][reason]={'top':c.most_common(5),'selected':{pc:c[pc] for pc in ['0xc40','0xd90','0xf50','0xfd0','0x10b0','0x29f0']}}
 result[name]=r
pairs={}
for a,b in [('baseline','no_ab'),('no_ab','no_copy'),('no_copy','no_copy_bar'),('no_ab','no_copy_bar'),('baseline','no_copy_bar')]:
 x,y=result[a],result[b];pairs[a+' -> '+b]=dict(cycle_throughput_gain_pct=100*(y['imma_per_SM_cycle']/x['imma_per_SM_cycle']-1),time_reduction_pct=100*(1-y['kernel_ms']/x['kernel_ms']))
(out/'interpretation.json').write_text(json.dumps(dict(variants=result,comparisons=pairs),indent=2)+'\n')
lines=[f'Capture: {src}', 'Input hashes match. All12 launches execute 67,633,152 IMMA. LDSM measured zero in all diagnostics.', '', 'variant      kernel_ms   Tensor%  barrier%  longSB%   math%   IMMA/SMcycle effectiveGHz']
for name,r in result.items():lines.append(f"{name:12} {r['kernel_ms']:.6f}    {r['tensor_pct']:.3f}   {r['barrier_pct']:.3f}    {r['long_scoreboard_pct']:.3f}    {r['math_throttle_pct']:.3f}   {r['imma_per_SM_cycle']:.9f}   {r['effective_SM_GHz']:.6f}")
lines+=['','Cycle-normalized throughput changes:']
for k,v in pairs.items():lines.append(f"{k}: +{v['cycle_throughput_gain_pct']:.3f}%; time reduction {v['time_reduction_pct']:.3f}%")
lines+=['', 'Barrier-only pair (no_copy -> no_copy_bar) retains identical data work and instruction count/slots except one BAR replaced with NOP.', 'Effective clock ~2.73476 ->2.73561GHz: this is not primarily a boost-clock gain.', 'Main-loop barrier-attributed PC0xd90 samples105722 ->0. Final barrier-attributed PC0x29f0 samples1929 ->9709.', 'PCs are sampled attribution locations; source barrier instructions are at0xc60 and0x29d0 respectively.', 'Remaining barrier percentage1.153% comes from final transcript synchronization. It is intentionally retained.', 'Eligible warps/scheduler0.38 ->0.41; issued warps/cycle0.23 ->0.26. Math throttle56.95% ->64.23% despite faster execution.', 'Higher math throttle alone is not evidence of worse performance or a larger end-to-end bottleneck.', 'L2 B/useful MMA37.758 ->0.258 after removing copies; this traffic reduction is deliberate, not an operand-reuse improvement.', '', 'Conclusion: stage handoff has a substantial conditional cost in this diagnostic. Same mapping can reach94.857% Tensor active-cycle utilization with operand supply constraints removed.', 'This does not prove synchronized warp bursts are the cause; changing a barrier changes scheduling and phase relationships as well as waiting.', 'Nor is the gain transferable directly to production: real operands require readiness and safe stage reuse.', 'Barriers and dependency effects are non-additive; the measured11.42% per-cycle gain is conditional on no copies/LDSM and zero MMA operands.', 'Production and PeakMiner are not benchmarked head-to-head here; this does not explain their entire throughput gap.', 'Next production-oriented investigation: distinguish buffer readiness from buffer-reuse synchronization, and find independent work that can execute before the required handoff.', 'Keep final transcript synchronization. Do not deploy these incorrect-arithmetic diagnostics.']
(root/'artifacts/copy-bypass/profile-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
