"""Static gates and optional event summaries for the current mapping layers."""
from pathlib import Path
import json,re
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-layers-current.sass').read_text();out={}
def instructions(sec):return re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)
for n in ('registers','shared','shared_sync','global','full'):
 sec=next(x for x in s.split('Function : ')[1:] if 'layers_current_'+n+'6kernel' in x.splitlines()[0]);rows=instructions(sec)
 out[n]={op:sum(ins.startswith(op) for pc,ins in rows) for op in ('IMMA','LDSM','LDGSTS','LDL','STL','BAR.SYNC','DEPBAR')}
 out[n]['IMMA_first_last']=[next(pc for pc,ins in rows if ins.startswith('IMMA')),next(pc for pc,ins in rows[::-1] if ins.startswith('IMMA'))]
 assert out[n]['IMMA']==64 and out[n]['LDL']==out[n]['STL']==0
old=next(x for x in (root/'artifacts/rank128-astream-ablation.sass').read_text().split('Function : ')[1:] if 'astream_ablation_full6kernel' in x.splitlines()[0])
new=next(x for x in s.split('Function : ')[1:] if 'layers_current_full6kernel' in x.splitlines()[0])
out['full_matches_measured_baseline']=instructions(old)==instructions(new)
assert out['full_matches_measured_baseline']
(root/'artifacts/rank128-layers-current-static.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
p=root/'artifacts/rank128-layers-current-events.json'
if p.exists():
 d=json.loads(p.read_text());print(json.dumps(d['summaries'],indent=2))
 vals=[d['summaries'][str(v)]['mean_ms'] for v in range(5)]
 deltas=[vals[i+1]-vals[i] for i in range(4)]
 report=dict(adjacent_differences_ms=deltas,full_vs_register_difference_ms=vals[-1]-vals[0],full_vs_register_time_ratio=vals[-1]/vals[0],caveat='Diagnostic deltas include code-generation, DVFS, and data-reuse changes; not additive causal instruction costs.')
 (root/'artifacts/rank128-layers-current-deltas.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
