"""Aggregate matched handoff PC hotspots with nearby machine instructions."""
from pathlib import Path
import json,re,collections
root=Path(__file__).resolve().parents[1];p=root/'artifacts/rank128-handoff-control-profile'
reports=json.loads((p/'pc-stalls.json').read_text());sass=(root/'artifacts/rank128-handoff-control.sass').read_text();out={}
for v,name in enumerate(('baseline','latecopy','earlywait')):
 rep=next(r for r in reports if Path(r['file']).name==f'v{v}.sass.csv')
 sec=next(s for s in sass.split('Function : ')[1:] if 'handoff_control_'+name+'6kernel' in s.splitlines()[0])
 rows=[dict(pc=hex(int(pc,16)),instruction=ins) for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)];lookup={r['pc']:i for i,r in enumerate(rows)}
 data={}
 for reason in ('stall_long_sb','stall_barrier','stall_short_sb','stall_math','stall_wait'):
  counts=collections.Counter()
  for row in rep['pcs']:counts[row['pc_offset']]+=row['samples'].get(reason,0)
  top=[]
  for pc,n in counts.most_common(5):
   if not n:continue
   i=lookup[pc];top.append(dict(pc=pc,samples=n,percent_of_reason=100*n/rep['sample_columns'][reason],context=rows[max(0,i-4):i+1]))
  data[reason]=dict(total=rep['sample_columns'][reason],top=top)
 out[name]=data
out['caveats']=['Sample shares are not wall-time fractions or removable costs.','Sample PC may follow the blocking instruction.','Address register overwrite is a candidate dependency, not proof of a particular WAR cause.']
(p/'hotspots.json').write_text(json.dumps(out,indent=2)+'\n')
for name in ('baseline','latecopy','earlywait'):
 print(name)
 for reason in ('stall_long_sb','stall_barrier'):
  print(reason,[(x['pc'],round(x['percent_of_reason'],2)) for x in out[name][reason]['top']])
issue=json.loads((p/'imma-issue.json').read_text())
for v in ('0','1','2'):
 x=issue[v]['means'];print(v,'IMMA/SMcycle',x['imma_per_SM_elapsed_cycle'],'cycles/IMMA/SMSP',1/x['imma_per_SMSP_elapsed_cycle'])
