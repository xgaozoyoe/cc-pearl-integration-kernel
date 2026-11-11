"""Baseline PC hotspots and nearby SASS; static dependency candidates, not causal proof."""
from pathlib import Path
import json,re,collections
root=Path(__file__).resolve().parents[1];profile=root/'artifacts/rank128-astream-ablation-profile'
rep=next(x for x in json.loads((profile/'pc-stalls.json').read_text()) if Path(x['file']).name=='v0.sass.csv')
sec=next(x for x in (root/'artifacts/rank128-astream-ablation.sass').read_text().split('Function : ')[1:] if 'astream_ablation_full' in x.splitlines()[0])
rows=[dict(pc=hex(int(pc,16)),instruction=ins) for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)]
lookup={x['pc']:i for i,x in enumerate(rows)};out={}
for reason in ('stall_long_sb','stall_short_sb','stall_wait','stall_math','stall_barrier'):
 counts=collections.defaultdict(float)
 for x in rep['pcs']:counts[x['pc_offset']]+=x['samples'].get(reason,0)
 top=[]
 for pc,n in sorted(counts.items(),key=lambda x:-x[1])[:8]:
  i=lookup[pc];top.append(dict(pc=pc,samples=n,percent_of_reason=100*n/rep['sample_columns'][reason],context=rows[max(0,i-5):i+1]))
 out[reason]=dict(total_samples=rep['sample_columns'][reason],top=top)
result=dict(source='Current A-stream baseline v0, capture2026-10-08; no new GPU profiling.',caveat='PC samples can be downstream; static preceding writers/readers do not prove the full dynamic dependency. Percentages are sampling shares,not removable wall time.',reasons=out,
 B_fragment_hotspots_share_of_short_sb=100*(14904+11816+9409+9170+6625)/73889,
 address_overwrite_hotspots_share_of_long_sb=100*(13405+9960)/28862)
(root/'artifacts/waited-results-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['B_fragment_hotspots_share_of_short_sb'],result['address_overwrite_hotspots_share_of_long_sb'])
