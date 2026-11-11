"""Verify/archive named-barrier capture and report throughput plus PC attribution."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,shutil,statistics
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);src=p.parse_args().capture.resolve()
for line in (src/'inputs.sha256').read_text().splitlines():
 digest,name=line.split(maxsplit=1);assert hashlib.sha256((root/name.strip()).read_bytes()).hexdigest()==digest,name
out=root/'artifacts/handoff-named-profile'
if out.exists():assert (out/'inputs.sha256').read_bytes()==(src/'inputs.sha256').read_bytes()
else:shutil.copytree(src,out)
s=json.loads((src/'summary.json').read_text());issues=json.loads((src/'imma-issue.json').read_text());pcs=json.loads((src/'pc-stalls.json').read_text());result={}
for v,name in [(0,'baseline'),(2,'late'),(3,'overlap')]:
 issue=issues[str(v)];assert len(issue['launches'])==3 and all(x['count_matches_sass'] and x['tensor_instructions']==67633152 for x in issue['launches'])
 m=issue['means'];metrics=s[str(v)]['metrics'];rep=next(x for x in pcs if Path(x['file']).name==f'v{v}.sass.csv')
 r=dict(kernel_ms=m['kernel_ms'],imma_per_SM_cycle=m['imma_per_SM_elapsed_cycle'],effective_SM_GHz=m['tensor_Ginst_sec']/m['imma_per_SM_elapsed_cycle']/128,l2_bytes_per_useful_mma=s[str(v)]['l2_bytes_per_useful_mma'])
 r.update({k:metrics[k]['mean'] for k in ['tensor_pct','l2_util_pct','barrier_pct','short_scoreboard_pct','long_scoreboard_pct','math_throttle_pct','eligible_warps_per_scheduler','issued_warps_per_cycle']})
 r['pc_samples']={}
 for reason in ['stall_barrier','stall_short_sb','stall_long_sb','stall_math','stall_wait','stall_membar']:
  c=collections.Counter();ins={}
  for x in rep['pcs']:c[x['pc_offset']]+=x['samples'].get(reason,0);ins[x['pc_offset']]=x['source']
  r['pc_samples'][reason]=dict(top=[dict(pc=pc,instruction=ins[pc],samples=n) for pc,n in c.most_common(5)],total=rep['sample_columns'].get(reason,0))
 r['math_samples_on_IMMA_pct']=100*sum(x['samples'].get('stall_math',0) for x in rep['pcs'] if x['opcode'].startswith('IMMA'))/rep['sample_columns']['stall_math']
 groups=[];header=None;base=None
 for row in csv.reader((src/f'v{v}.sass.csv').open()):
  if row and row[0]=='Kernel Name':g={};groups.append(g);base=None
  elif row and row[0]=='Address':header=row
  elif row and row[0].startswith('0x'):
   if base is None:base=int(row[0],16)
   d=dict(zip(header,row));g[int(row[0],16)-base]=(d['Source'].strip(),int(d['Instructions Executed'].replace(',','')),base)
 assert len(groups)==3
 r['dynamic_instructions']=statistics.mean(sum(x[1] for x in g.values()) for g in groups)
 ops=collections.Counter();zones=collections.Counter()
 for g in groups:
  mma=[pc for pc,(i,n,b) in g.items() if i.startswith('IMMA')]
  back,branch=next((pc,i) for pc,(i,n,b) in g.items() if pc>max(mma) and i.startswith('BRA ') and int(i.split()[-1],16)-b<min(mma))
  start=int(branch.split()[-1],16)-next(iter(g.values()))[2]
  for pc,(ins,n,b) in g.items():
   w=ins.split();op=w[1] if w[0].startswith('@') else w[0];ops[op]+=n/3
   zone='prologue' if pc<start else 'header/copy' if pc<min(mma) else 'MMA/operand' if pc<=max(mma) else 'fold/tail' if pc<=back else 'output'
   zones[zone]+=n/3
 r['instructions_by_opcode']=dict(ops);r['instructions_by_region']=dict(zones);result[name]=r
b,l,o=[result[n] for n in ['baseline','late','overlap']]
lines=[f'Capture:{src}', 'All input hashes match current experiment;3 launches/variant;67,633,152 executed IMMA each.', '', 'variant kernel_ms Tensor% IMMA/SMcycle barrier% shortSB% longSB% math% L2B/usefulMMA']
for n,r in result.items():lines.append(f"{n:9} {r['kernel_ms']:.6f} {r['tensor_pct']:.3f} {r['imma_per_SM_cycle']:.9f} {r['barrier_pct']:.3f} {r['short_scoreboard_pct']:.3f} {r['long_scoreboard_pct']:.3f} {r['math_throttle_pct']:.3f} {r['l2_bytes_per_useful_mma']:.6f}")
lines+=['',f"Overlap IMMA/cycle vs baseline:{100*(o['imma_per_SM_cycle']/b['imma_per_SM_cycle']-1):+.4f}%; vs late:{100*(o['imma_per_SM_cycle']/l['imma_per_SM_cycle']-1):+.4f}%.",f"Overlap kernel time vs baseline:{100*(o['kernel_ms']/b['kernel_ms']-1):+.4f}%.",f"Effective SM GHz baseline/late/overlap:{b['effective_SM_GHz']:.6f}/{l['effective_SM_GHz']:.6f}/{o['effective_SM_GHz']:.6f}.",f"Dynamic instructions baseline/late/overlap:{b['dynamic_instructions']:.0f}/{l['dynamic_instructions']:.0f}/{o['dynamic_instructions']:.0f}.",f"Overlap instructions vs baseline:{100*(o['dynamic_instructions']/b['dynamic_instructions']-1):+.4f}% (old spin capture was+28%).",'', 'Dynamic opcode deltas, overlap minus baseline (warp instruction executions/kernel):']
delta={op:o['instructions_by_opcode'].get(op,0)-b['instructions_by_opcode'].get(op,0) for op in set(o['instructions_by_opcode'])|set(b['instructions_by_opcode'])}
for op,n in sorted(delta.items(),key=lambda x:abs(x[1]),reverse=True)[:12]:lines.append(f'{op:25} {n:+.0f}')
lines+=['', 'Region boundaries are first/last IMMA and main loop backedge; compiler motion can move work across them:']
for zone in b['instructions_by_region']:lines.append(f"{zone:15} {b['instructions_by_region'][zone]:.0f} -> {o['instructions_by_region'][zone]:.0f}")
lines+=['', 'PC attribution:']
for n,r in result.items():
 lines.append(n+f": math samples on IMMA={r['math_samples_on_IMMA_pct']:.3f}%")
 for reason in ['stall_barrier','stall_short_sb','stall_long_sb','stall_math']:
  lines.append(reason+': '+str(r['pc_samples'][reason]['top'][:2]))
lines+=['', 'Interpretation:',
'1. Earlier arrival improves the named-barrier design relative to its late control, but NOT relative to production baseline.',
'2. Barrier stall falls12.79->8.95%; math throttle rises45.39->49.06%. These are scheduler/sample metrics, not additive wall-time costs.',
'3. Useful IMMA/cycle and Tensor utilization do not rise relative to baseline. Do not claim improved Tensor density versus production.',
'4. Named overlap has no shared readiness polling loop; operand LDS/LDSM are still present. L2 bytes/usefulMMA and occupancy are preserved.',
'5. Extra instructions are spread across loop header/copy, operand/MMA and tail. Do not attribute the full+6.7% solely to barrier ID arithmetic.',
'6. In overlap, DEPBAR1690 precedes IMMA16a0. The following PC has81559 math-throttle samples; this is not evidence for81559 copy-wait samples.',
'7. Main receiving barrier stalls are sampled at2a80 after BAR.SYNC2a60:89121/90188 barrier samples. No new dominant long/short scoreboard spike.',
'8. Higher ready-warps (.32->.37) and issued-warps (.22->.23) coexist with flat useful MMA throughput; this does not prove phase bunching causally.',
'Next bounded experiment: reduce recurring control/address work while preserving this validated handoff, then repeat matched IMMA/cycle and full-grid timing.',
'Check barrier-ID invariants and operand-address recalculation in SASS first; do not add artificial warp delay or claim math-throttle reduction alone is a win.',
'Previous offline full grids:baseline308.535ms/228.074TH/s;overlap309.763ms/227.169TH/s. No production promotion.',
'No kernel or production changes in this analysis.']
(out/'interpretation.json').write_text(json.dumps(result,indent=2)+'\n');(root/'artifacts/handoff-named-profile-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:17]))
