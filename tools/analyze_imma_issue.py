"""Direct elapsed-SM-cycle normalization for profile_imma_issue.sh captures."""
import csv,json,statistics,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--variants',type=int,nargs='+',default=[0,1,4]);a=p.parse_args();out={}
for v in a.variants:
 rows=list(csv.reader((a.directory/f'v{v}.raw.csv').open()));i=next(i for i,r in enumerate(rows) if 'ID' in r and 'Kernel Name' in r);h=rows[i];values=[]
 for row in rows[i+1:]:
  if len(row)!=len(h):continue
  d=dict(zip(h,row))
  try:
   n=float(d['smsp__inst_executed_pipe_tensor_v2.sum'].replace(',',''));sm=float(d['sm__cycles_elapsed.sum'].replace(',',''));sp=float(d['smsp__cycles_elapsed.sum'].replace(',',''));ns=float(d['gpu__time_duration.sum'].replace(',',''))
  except ValueError:continue
  values.append(dict(tensor_instructions=n,expected_sass_IMMA=67633152,count_matches_sass=n==67633152,imma_per_SM_elapsed_cycle=n/sm,imma_per_SMSP_elapsed_cycle=n/sp,tensor_Ginst_sec=n/ns,kernel_ms=ns/1e6))
 out[v]=dict(launches=values,means={k:statistics.mean(x[k] for x in values) for k in ('imma_per_SM_elapsed_cycle','imma_per_SMSP_elapsed_cycle','tensor_Ginst_sec','kernel_ms')})
(a.directory/'imma-issue.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
