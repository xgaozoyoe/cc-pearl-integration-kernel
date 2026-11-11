"""Verify/archive copy-cursor capture and report throughput plus PC attribution."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,shutil,statistics
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);src=p.parse_args().capture.resolve()
for line in (src/'inputs.sha256').read_text().splitlines():
 digest,name=line.split(maxsplit=1);assert hashlib.sha256((root/name.strip()).read_bytes()).hexdigest()==digest,name
out=root/'artifacts/handoff-cursor-profile'
if out.exists():assert (out/'inputs.sha256').read_bytes()==(src/'inputs.sha256').read_bytes()
else:shutil.copytree(src,out)
s=json.loads((src/'summary.json').read_text());issues=json.loads((src/'imma-issue.json').read_text());pcs=json.loads((src/'pc-stalls.json').read_text());result={}
for v,name in [(0,'baseline'),(1,'control'),(2,'hoist'),(3,'induction')]:
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

(out/'interpretation.json').write_text(json.dumps(result,indent=2)+'\n')
for name,r in result.items():
 print(name,{k:r[k] for k in ['kernel_ms','tensor_pct','imma_per_SM_cycle','effective_SM_GHz','dynamic_instructions','barrier_pct','short_scoreboard_pct','long_scoreboard_pct','math_throttle_pct','l2_bytes_per_useful_mma']})
 for reason in ['stall_barrier','stall_short_sb','stall_long_sb','stall_math']:print(reason,r['pc_samples'][reason]['top'][:2])
c=result['control']
lines=[f'Capture:{src}', 'All source/library/harness hashes verified;3 launches per variant,67,633,152 IMMA each.', '', 'variant kernel_ms Tensor% IMMA/SMcycle dynamic_inst barrier% shortSB% longSB% math%']
for name,r in result.items():lines.append(f"{name:10} {r['kernel_ms']:.6f} {r['tensor_pct']:.3f} {r['imma_per_SM_cycle']:.9f} {r['dynamic_instructions']:.0f} {r['barrier_pct']:.3f} {r['short_scoreboard_pct']:.3f} {r['long_scoreboard_pct']:.3f} {r['math_throttle_pct']:.3f}")
for name in ['hoist','induction']:
 r=result[name];lines+=['',name+f": dynamic instructions vs control={100*(r['dynamic_instructions']/c['dynamic_instructions']-1):+.3f}%; IMMA/cycle={100*(r['imma_per_SM_cycle']/c['imma_per_SM_cycle']-1):+.3f}%; time={100*(r['kernel_ms']/c['kernel_ms']-1):+.3f}%.",f"Effective SM GHz control/candidate={c['effective_SM_GHz']:.6f}/{r['effective_SM_GHz']:.6f}.",f"L2 bytes/usefulMMA control/candidate={c['l2_bytes_per_useful_mma']:.6f}/{r['l2_bytes_per_useful_mma']:.6f}."]
 d={op:r['instructions_by_opcode'].get(op,0)-c['instructions_by_opcode'].get(op,0) for op in set(r['instructions_by_opcode'])|set(c['instructions_by_opcode'])}
 lines.append('Largest opcode count changes:'+str(sorted(d.items(),key=lambda x:abs(x[1]),reverse=True)[:8]))
 for reason in ['stall_short_sb','stall_long_sb','stall_math','stall_barrier']:lines.append(reason+': '+str(r['pc_samples'][reason]['top'][:2]))
lines+=['', 'Findings:',
'1. Copy-address simplification reduces actual executed instructions, not merely static code size. Neither candidate improves performance.',
'2. Regression persists after frequency normalization; candidates had higher effective clock yet lower IMMA/cycle.',
'3. Same MMA count,L2 bytes/usefulMMA,168regs,0spills,69632Bshared,1CTA/SM. No evidence of additional traffic or lost occupancy.',
'4. Barrier-stall share decreases while short-scoreboard and math-throttle shares increase. These shares are not additive wall-time costs.',
'5. Important short-scoreboard samples occur at IMMA operand consumers. E.g.induction1270 immediately follows B LDSM1260;hoist1360 immediately follows B LDSM1350. Control also has analogous immediate consumers; adjacency alone is not a newly introduced defect.',
'6. SASS scheduling changed: copy wait moved from after48IMMA to after56IMMA. Source arithmetic simplification is not an isolated scheduling intervention.',
'7. Operand supply/scheduling is implicated; attribution of all slowdown to a particular LDSM,wait or register dependency is NOT established. Extra scalar instructions may overlap latency, but that explanation remains a hypothesis.',
'Decision:retain validated lean checkpoint;do not promote hoist/induction. Stay with handoff overlap, but next reductions should preserve operand schedule or explicitly control it.',
'Next bounded diagnostic:choose a hot LDSM-consumer dependency and alter only its load-to-use spacing on the lean checkpoint while keeping handoff/copy work fixed;inspect SASS before benchmarking. Do not infer benefit from instruction-count reduction alone.',
'No production modifications or GPU execution during this analysis.']
(root/'artifacts/handoff-cursor-profile-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:14]))
