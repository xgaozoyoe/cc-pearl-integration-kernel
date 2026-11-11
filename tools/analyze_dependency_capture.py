"""Cycle-normalized interpretation of baseline operand-substitution probes."""
from pathlib import Path
import json,collections
root=Path(__file__).resolve().parents[1];p=root/'artifacts/dependency-ablation/profile'
summary=json.loads((p/'summary.json').read_text());issue=json.loads((p/'imma-issue.json').read_text());pc=json.loads((p/'pc-stalls.json').read_text())
names=['baseline','c_zero','a_zero','b_zero','ab_zero'];base=issue['0']['means']['imma_per_SM_elapsed_cycle'];result={}
for v,name in enumerate(names):
 im=issue[str(v)];assert all(x['count_matches_sass'] for x in im['launches'])
 means=im['means'];m=summary[str(v)]['metrics']
 row=dict(kernel_ms=means['kernel_ms'],IMMA_per_SM_cycle=means['imma_per_SM_elapsed_cycle'],cycle_throughput_change_pct=100*(means['imma_per_SM_elapsed_cycle']/base-1),tensor_pct=m['tensor_pct']['mean'],short_scoreboard_pct=m['short_scoreboard_pct']['mean'],long_scoreboard_pct=m['long_scoreboard_pct']['mean'],barrier_pct=m['barrier_pct']['mean'],math_throttle_pct=m['math_throttle_pct']['mean'])
 # n/time divided by n/sum(SM cycles) = sum(SMcycles)/time; divide128SMs.
 row['aggregate_effective_SM_GHz']=means['tensor_Ginst_sec']/means['imma_per_SM_elapsed_cycle']/128
 rep=next(x for x in pc if Path(x['file']).name==f'v{v}.sass.csv')
 sample=next(x for x in rep['pcs'] if x['pc_offset']=='0x11a0')
 row['hotspot_IMMA_instruction']=sample['source']
 row['hotspots']={}
 for reason in ('stall_short_sb','stall_long_sb','stall_math'):
  c=collections.Counter()
  for x in rep['pcs']:c[x['pc_offset']]+=x['samples'].get(reason,0)
  row['hotspots'][reason]=[dict(pc=k,samples=n,share_of_reason_pct=100*n/rep['sample_columns'][reason]) for k,n in c.most_common(5)]
 result[name]=row
(p/'interpretation.json').write_text(json.dumps(result,indent=2)+'\n')
for name,r in result.items():print(name,{k:v for k,v in r.items() if k!='hotspots'})
