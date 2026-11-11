"""Same-machine-code transcript on/off attribution; samples are not wall time."""
from pathlib import Path
import argparse,csv,json,collections
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
reports=json.loads((d/'pc-stalls.json').read_text());out={}
for rep in reports:
 v=int(Path(rep['file']).name.split('.')[0][1:]);lo,hi=(0x1790,0x29a0) if v==0 else (0x17a0,0x29b0)
 endstart=0x29d0 if v==0 else 0x29e0
 phases=collections.defaultdict(lambda:collections.defaultdict(float));short=collections.defaultdict(float);top=collections.defaultdict(float);source={}
 for row in rep['pcs']:
  pc=int(row['pc_offset'],16);phase='fold_region' if lo<=pc<hi else 'epilogue' if pc>=endstart else 'main_and_setup'
  for k,n in row['samples'].items():
   if 'Not' not in k:phases[phase][k]+=n
  if phase=='fold_region':
   short[row['opcode']]+=row['samples'].get('stall_short_sb',0)
   top[row['pc_offset']]+=row['samples'].get('stall_short_sb',0);source[row['pc_offset']]=row['source']
 total=rep['sample_columns']['# Samples']
 # Dynamic instruction counts from source export (including all 3 launches).
 count=collections.defaultdict(float);header=None;base=None
 with (d/f'v{v}.sass.csv').open() as f:
  for row in csv.reader(f):
   if not row:continue
   if row[0]=='Kernel Name':header=None;base=None;continue
   if 'Address' in row and 'Source' in row:header=row;continue
   if header is None or len(row)!=len(header):continue
   fields=dict(zip(header,row))
   try:addr=int(fields['Address'],16)
   except ValueError:continue
   if base is None:base=addr
   pc=addr-base;phase='fold_region' if lo<=pc<hi else 'epilogue' if pc>=endstart else 'main_and_setup'
   try:count[phase]+=float(fields['Instructions Executed'].replace(',',''))
   except ValueError:pass
 out[v]=dict(total_samples=total,phases={k:dict(samples=x['# Samples'],sample_share_pct=100*x['# Samples']/total,short_sb=x['stall_short_sb'],wait=x['stall_wait'],dynamic_warp_instructions=count[k]) for k,x in phases.items()},fold_short_by_opcode=dict(short),fold_top_short=[dict(pc=pc,samples=n,instruction=source[pc]) for pc,n in sorted(top.items(),key=lambda x:-x[1])[:15]])
summary=json.loads((d/'summary.json').read_text());on=summary['4']['metrics']['kernel_ns']['mean'];off=summary['1']['metrics']['kernel_ns']['mean']
result=dict(note='PC sample shares are not wall-clock time fractions. A skipped instruction can have attributed samples (PC attribution skew); consult dynamic instruction counts. Phase bounds verified against current SASS only.',kernel_delta_ms=(on-off)/1e6,kernel_delta_pct=100*(on-off)/on,variants=out)
(d/'attribution.json').write_text(json.dumps(result,indent=2)+'\n')
for v,x in out.items():print(v,x['phases'],x['fold_short_by_opcode'])
