"""Compare exact-hot-loop group memberships and attribute PC samples without claiming instruction latency."""
from pathlib import Path
import json,argparse,collections
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args();issue=json.loads((a.capture/'imma-issue.json').read_text());pcs=json.loads((a.capture/'pc-stalls.json').read_text())
names={'0':'native','1':'masked_control','3':'rows6'};lines=['Same frozen hot loop, GPU0 only. NCU base clock policy; compare elapsed-cycle normalized IMMA rate.', 'PC samples are sampled attribution, not additive instruction runtimes. A stalled PC may wait for a preceding deferred barrier.', 'Late named sync0x2a00 typically blocks at0x2a30; early arrival0x16d0 also waits for final LDSM completion.']
base=issue['1']['means']['imma_per_SM_elapsed_cycle']
for key,name in names.items():
 v=issue[key];assert v['launches'] and all(x['count_matches_sass'] for x in v['launches'])
 lines.append(name+' '+json.dumps(v['means'])+f' IMMA/cycle vs matched control {(v["means"]["imma_per_SM_elapsed_cycle"]/base-1)*100:+.4f}%')
 data=next(x for x in pcs if Path(x['file']).name==f'v{key}.sass.csv')
 for metric in ['stall_barrier','stall_short_sb','stall_long_sb','stall_math','stall_wait']:
  totals=collections.defaultdict(float);sources={}
  for r in data['pcs']:
   totals[r['pc_offset']]+=r['samples'].get(metric,0);sources[r['pc_offset']]=r['source']
  total=sum(totals.values());lines.append(f'{metric}: total={total:g}')
  for pc,count in sorted(totals.items(),key=lambda x:-x[1])[:6]:
   if count:lines.append(f'  {pc} {count:g} ({100*count/total:.2f}% of {metric}) {sources[pc]}')
lines+=['Interpretation checklist: confirm native vs initialization control agree; locate additional rows6 barrier samples; compare eligible warps and IMMA/cycle.', 'Do not infer scheduler-to-warp assignment or which individual warp arrived last from aggregate PC sampling.']
(a.capture/'handoff-membership-profile-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
