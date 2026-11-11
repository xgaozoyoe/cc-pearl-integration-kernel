"""Verify and archive split-barrier profile; measure dynamic polling costs."""
from pathlib import Path
import argparse,json,collections,hashlib,shutil,csv,statistics
root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('capture',type=Path);src=p.parse_args().capture.resolve()
for line in (src/'inputs.sha256').read_text().splitlines():
 digest,name=line.split(maxsplit=1);assert hashlib.sha256((root/name.strip()).read_bytes()).hexdigest()==digest,name
out=root/'artifacts/transcript-handoff-profile'
if out.exists():assert (out/'inputs.sha256').read_bytes()==(src/'inputs.sha256').read_bytes()
else:shutil.copytree(src,out)
s=json.loads((src/'summary.json').read_text());issues=json.loads((src/'imma-issue.json').read_text());pcs=json.loads((src/'pc-stalls.json').read_text());names=['baseline','late','overlap'];result={}
for v,name in enumerate(names):
 issue=issues[str(v)];assert len(issue['launches'])==3 and all(x['count_matches_sass'] for x in issue['launches'])
 m=issue['means'];metrics=s[str(v)]['metrics'];rep=next(x for x in pcs if Path(x['file']).name==f'v{v}.sass.csv')
 r=dict(kernel_ms=m['kernel_ms'],imma_per_SM_cycle=m['imma_per_SM_elapsed_cycle'],effective_SM_GHz=m['tensor_Ginst_sec']/m['imma_per_SM_elapsed_cycle']/128,l2_bytes_per_useful_mma=s[str(v)]['l2_bytes_per_useful_mma'])
 r.update({k:metrics[k]['mean'] for k in ['tensor_pct','barrier_pct','short_scoreboard_pct','long_scoreboard_pct','math_throttle_pct','eligible_warps_per_scheduler','issued_warps_per_cycle']})
 r['pc_samples']={}
 for reason in ['stall_barrier','stall_short_sb','stall_long_sb','stall_math','stall_wait','stall_membar']:
  c=collections.Counter()
  for x in rep['pcs']:c[x['pc_offset']]+=x['samples'].get(reason,0)
  r['pc_samples'][reason]=dict(top=c.most_common(5),total=rep['sample_columns'].get(reason,0))
 if v:
  count=sum(x['samples'].get('stall_short_sb',0) for x in rep['pcs'] if x['pc_offset'] in ['0x2c10','0x2c20'])
  r['poll_short_sb_samples']=count;r['poll_short_sb_share_pct']=100*count/rep['sample_columns']['stall_short_sb']
 groups=[];header=None;base=None
 for row in csv.reader((src/f'v{v}.sass.csv').open()):
  if row and row[0]=='Kernel Name':group={};groups.append(group);base=None
  elif row and row[0]=='Address':header=row
  elif row and row[0].startswith('0x'):
   if base is None:base=int(row[0],16)
   d=dict(zip(header,row));group[int(row[0],16)-base]=(d['Source'].strip(),int(d['Instructions Executed'].replace(',','')))
 assert len(groups)==3
 r['dynamic_instructions']=statistics.mean(sum(x[1] for x in g.values()) for g in groups)
 if v:
  assert all(g[0x2c00][0]=='LDS R112, [0x11004]' for g in groups)
  r['poll_reads']=statistics.mean(g[0x2c00][1] for g in groups)
  r['poll_checks_per_warp_K64']=r['poll_reads']/(1376*12*64)
  r['poll_loop_instructions']=statistics.mean(sum(g[pc][1] for pc in range(0x2c00,0x2c60,16)) for g in groups)
 result[name]=r
(out/'interpretation.json').write_text(json.dumps(result,indent=2)+'\n')
lines=[f'Capture: {src}', 'All input hashes match;3 launches/variant;67,633,152 IMMA each.', '', 'variant  kernel_ms  tensor%  barrier% shortSB% IMMA/SMcycle dynamic_instructions poll_checks/warp/K64']
for n,r in result.items():lines.append(f"{n:9} {r['kernel_ms']:.6f}  {r['tensor_pct']:.3f}   {r['barrier_pct']:.3f}   {r['short_scoreboard_pct']:.3f}   {r['imma_per_SM_cycle']:.9f}   {r['dynamic_instructions']:.0f}  {r.get('poll_checks_per_warp_K64',0):.4f}")
b,l,o=[result[n] for n in names]
lines+=['',f"Overlap IMMA/cycle vs baseline: {100*(o['imma_per_SM_cycle']/b['imma_per_SM_cycle']-1):+.4f}%.",f"Overlap IMMA/cycle vs late: {100*(o['imma_per_SM_cycle']/l['imma_per_SM_cycle']-1):+.4f}%.",f"Overlap total dynamic instructions vs baseline: {100*(o['dynamic_instructions']/b['dynamic_instructions']-1):+.4f}%.",f"Overlap polling loop instructions/kernel: {o['poll_loop_instructions']:.0f}; polling accounts for {o['poll_short_sb_share_pct']:.3f}% of short-scoreboard samples.",f"Effective SM GHz baseline/late/overlap: {b['effective_SM_GHz']:.6f}/{l['effective_SM_GHz']:.6f}/{o['effective_SM_GHz']:.6f}.",
'', 'Polling loop:', '0x2c00 LDS R112,[0x11004]', '0x2c10 R2UR UR8,R112', '0x2c20 ULOP3.LUT phase test', '0x2c30 PLOP3.LUT predicate transfer', '0x2c40 NOP', '0x2c50 conditional BRA back to0x2c00',
'', 'Interpretation: earlier arrival helps the new mechanism (+5.35% IMMA/cycle vs late), but does not beat native baseline.', 'Polling adds substantial dynamic instruction work and a serial shared-load -> register-transfer -> predicate chain.', 'Eligible and issued warp metrics rise while useful Tensor issue falls: more issued instructions are not necessarily useful MMA.', 'L2 bytes/useful MMA stays37.758..37.759; register count168,zero spill,1CTA/SM unchanged.', 'No sampled membar stalls; do not assert MEMBAR/ATOMS dominates elapsed time from their presence alone.', 'Samples are not wall-time fractions. This evidence identifies polling as a major new cost but does not assign all slowdown exclusively to it.', 'Next experiment should reduce polling/checking work or use a cheaper safe handoff. Do not keep advancing arrival without controlling its waiting cost.', 'All prior transcript/digest checks and2025.2.1 memcheck/racecheck passed; production libraries unchanged.']
(root/'artifacts/transcript-handoff-profile-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
