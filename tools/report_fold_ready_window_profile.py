"""Read actual counters for the frozen-SASS local XOR wait-boundary comparison."""
from pathlib import Path
import json,argparse,collections
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args()
issue=json.loads((a.capture/'imma-issue.json').read_text());summary=json.loads((a.capture/'summary.json').read_text());pcs=json.loads((a.capture/'pc-stalls.json').read_text())
lines=['GPU0,3profiled launches/variant;NCU base clock policy. Region8192x8192,not full grids.',
'Eligible/issued warps are rounded to0.01 in the exported metrics;equal printed values do not prove exact equality.',
'Compare SM elapsed-cycle normalized IMMA rate rather than small wall-time differences.',
'name kernel_ms IMMA/SMcycle tensor_pct eligible issued barrier_pct short_SB_pct math_throttle_pct L2bytes/usefulMMA']
names={'0':'native','1':'after_wait','2':'before_wait'}
for key,name in names.items():
 v=issue[key];assert len(v['launches'])==3 and all(x['count_matches_sass'] for x in v['launches'])
 z=summary[key]['metrics'];q=v['means'];get=lambda k:z[k]['mean']
 lines.append(f"{name} {q['kernel_ms']:.6f} {q['imma_per_SM_elapsed_cycle']:.9f} {get('tensor_pct'):.4f} {get('eligible_warps_per_scheduler'):.2f} {get('issued_warps_per_cycle'):.2f} {get('barrier_pct'):.4f} {get('short_scoreboard_pct'):.4f} {get('math_throttle_pct'):.4f} {summary[key]['l2_bytes_per_useful_mma']:.6f}")
for base in ('0','1'):
 lines.append(f"before_wait vs {names[base]}:IMMA/cycle {(issue['2']['means']['imma_per_SM_elapsed_cycle']/issue[base]['means']['imma_per_SM_elapsed_cycle']-1)*100:+.5f}%;walltime {(issue['2']['means']['kernel_ms']/issue[base]['means']['kernel_ms']-1)*100:+.5f}%")
for key,name in names.items():
 d=next(d for d in pcs if Path(d['file']).name==f'v{key}.sass.csv');lines.append(name+' PC sampling (not instruction latency):')
 for metric in ('stall_barrier','stall_short_sb','stall_math','stall_wait'):
  counts=collections.defaultdict(float);source={}
  for r in d['pcs']:counts[r['pc_offset']]+=r['samples'].get(metric,0);source[r['pc_offset']]=r['source']
  lines.append(metric+' total='+str(sum(counts.values())))
  for pc,n in sorted(counts.items(),key=lambda x:-x[1])[:3]:lines.append(f' {pc} {n:g} {source[pc]}')
 lines.append('window PCs:pc all_samples short_SB wait branch_resolving no_inst')
 for pc in ('0x16c0','0x16d0','0x3780','0x3790','0x37a0','0x37b0','0x37c0'):
  rows=[r for r in d['pcs'] if r['pc_offset']==pc]
  if rows:lines.append(pc+' '+' '.join(str(sum(r['samples'].get(k,0) for r in rows)) for k in ('# Samples','stall_short_sb','stall_wait','stall_branch_resolving','stall_no_inst'))+' '+rows[0]['source'])
lines+=['Conclusion:no measured increase in eligible warps at reported precision;no material IMMA/cycle benefit versus matched control.',
'Short-scoreboard reduction versus native occurs in BOTH relocated variants,so it is not evidence of a before-wait scheduling benefit.',
'PCs retain main named-barrier waiting at2a30;local code motion does not remove that synchronization.',
'Only3local XORs of1block moved,not REDUX/stores or the full16-block schedule;this result cannot falsify all fine-grained transcript scheduling.',
'Extra2branches/iteration,3NOPs/fold,local6cycle stalls and extra nofold XOR issue remain confounds versus native.',
'No production changes,no new GPU runs.']
(a.capture/'interpretation.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
