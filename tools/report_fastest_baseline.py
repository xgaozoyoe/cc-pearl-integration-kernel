"""Current binary PC attribution. Sample fractions are not removable wall time."""
from pathlib import Path
import argparse,json,collections,csv,statistics
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args();out=a.capture
profiles=json.loads((out/'pc-stalls.json').read_text());summary=json.loads((out/'summary.json').read_text());issue=json.loads((out/'imma-issue.json').read_text())
def region(pc):
 if pc<0xbb0:return 'prologue'
 if pc<0xdd0:return 'stage_address_setup'
 if pc<0x1090:return 'copy_address_and_issue'
 if pc<0x1120:return 'first_loads_and_first_mma'
 if pc<0x16c0:return 'interior_ldsm_mma'
 if pc<0x16e0:return 'copy_wait_and_early_arrival'
 if pc<0x17e0:return 'tail_mma_after_arrival'
 if pc<0x2a00:return 'transcript_and_gate'
 if pc<0x2a50:return 'stage_sync_and_backedge'
 return 'output_epilogue'
lines=['Exact fastest both_compact binary, no source recompilation.', 'Two captures of SAME variant; fixed profiling base clocks, application replay, warm cache policy none.', 'Kernel times under profiler are not full-grid wall times or pool TH/s.', 'Stall samples identify where a warp was sampled waiting; they do not measure additive execution cost.', 'Deferred BAR.SYNC stalls may appear on subsequent instructions; stage_sync_and_backedge groups those PCs.', 'first_loads window is a static PC range, not a temporal phase across all warps.', 'No inference of simultaneous warp phases from aggregate samples.', ''];results=[]
for i,profile in enumerate(profiles):
 key=str(i);s=summary[key];lines.append(f'CAPTURE{i}: '+json.dumps(issue[key]['means']))
 lines.append('Metrics: '+json.dumps({k:v.get('mean') for k,v in s['metrics'].items()}))
 raw=list(csv.reader((out/f'v{i}.raw.csv').open()));hi=next(j for j,r in enumerate(raw) if 'ID' in r and 'Kernel Name' in r);header=raw[hi];extra={}
 for metric in ('dram__bytes.sum','sm__sass_data_bytes_mem_shared_op_ldsm.sum','smsp__inst_executed_pipe_alu.sum','smsp__inst_executed_pipe_tensor_v2.sum','sm__cycles_elapsed.avg.per_second'):
  vals=[]
  for row in raw[hi+1:]:
   if len(row)!=len(header):continue
   d=dict(zip(header,row))
   if not d.get('ID','').isdigit():continue
   try:vals.append(float(d[metric].replace(',','')))
   except (ValueError,KeyError):continue
  if vals:extra[metric]=statistics.mean(vals)
 lines.append('Extra raw metrics: '+json.dumps(extra))
 lines.append('L2 bytes/useful MMA: '+str(s.get('l2_bytes_per_useful_mma')))

 groups=collections.defaultdict(lambda:collections.defaultdict(float));pcs=collections.defaultdict(lambda:collections.defaultdict(float));ins={}
 for r in profile['pcs']:
  pc=int(r['pc_offset'],16);ins[pc]=r['source']
  for col,value in r['samples'].items():groups[region(pc)][col]+=value;pcs[pc][col]+=value
 result={'capture':i,'groups':{k:dict(v) for k,v in groups.items()},'columns':{}}
 for col,total in profile['sample_columns'].items():
  if not total:continue
  ranked=sorted(((g,v.get(col,0)) for g,v in groups.items()),key=lambda x:-x[1])
  top=sorted(((pc,v.get(col,0)) for pc,v in pcs.items()),key=lambda x:-x[1])[:10]
  result['columns'][col]={'total':total,'regions_percent':{g:100*v/total for g,v in ranked},'top_pcs':[{'pc':hex(pc),'samples':v,'percent':100*v/total,'instruction':ins[pc]} for pc,v in top if v]}
  if any(w in col.lower() for w in ('barrier','scoreboard','short_sb','long_sb','stall_math','stall_mio','wait','not_selected')):
   lines.append(col+': '+', '.join(f'{g}={100*v/total:.2f}%' for g,v in ranked if v))
   lines.extend(f'  {hex(pc)} {100*v/total:.2f}% {ins[pc]}' for pc,v in top[:5] if v)
 results.append(result);lines.append('')
(out/'phase-attribution.json').write_text(json.dumps(results,indent=2)+'\n');(out/'fastest-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
