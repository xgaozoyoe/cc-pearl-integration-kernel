"""Verify/archive lean handoff capture and report throughput plus PC attribution."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,shutil,statistics
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('capture',type=Path);src=p.parse_args().capture.resolve()
for line in (src/'inputs.sha256').read_text().splitlines():
 digest,name=line.split(maxsplit=1);assert hashlib.sha256((root/name.strip()).read_bytes()).hexdigest()==digest,name
out=root/'artifacts/handoff-lean-profile'
if out.exists():assert (out/'inputs.sha256').read_bytes()==(src/'inputs.sha256').read_bytes()
else:shutil.copytree(src,out)
s=json.loads((src/'summary.json').read_text());issues=json.loads((src/'imma-issue.json').read_text());pcs=json.loads((src/'pc-stalls.json').read_text());result={}
for v,name in [(0,'baseline'),(1,'named'),(3,'toggle')]:
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
b,n,t=[result[k] for k in ['baseline','named','toggle']]
lines=[f'Capture:{src}', 'All input hashes match;3 launches per variant,67,633,152 IMMA executed each.', '', 'variant kernel_ms Tensor% IMMA/SMcycle barrier% shortSB% longSB% math% dynamic_instructions']
for name,r in result.items():lines.append(f"{name:9} {r['kernel_ms']:.6f} {r['tensor_pct']:.3f} {r['imma_per_SM_cycle']:.9f} {r['barrier_pct']:.3f} {r['short_scoreboard_pct']:.3f} {r['long_scoreboard_pct']:.3f} {r['math_throttle_pct']:.3f} {r['dynamic_instructions']:.0f}")
lines+=['',f"Toggle IMMA/cycle vs baseline:{100*(t['imma_per_SM_cycle']/b['imma_per_SM_cycle']-1):+.4f}%; vs previous named:{100*(t['imma_per_SM_cycle']/n['imma_per_SM_cycle']-1):+.4f}%.",f"Toggle kernel time vs baseline:{100*(t['kernel_ms']/b['kernel_ms']-1):+.4f}%.",f"Effective SM GHz baseline/named/toggle:{b['effective_SM_GHz']:.6f}/{n['effective_SM_GHz']:.6f}/{t['effective_SM_GHz']:.6f}.",f"Toggle dynamic instructions vs baseline:{100*(t['dynamic_instructions']/b['dynamic_instructions']-1):+.4f}%; vs named:{100*(t['dynamic_instructions']/n['dynamic_instructions']-1):+.4f}%.", '', 'Dynamic opcode deltas: toggle minus previous named (warp instruction executions/kernel).']
delta={op:t['instructions_by_opcode'].get(op,0)-n['instructions_by_opcode'].get(op,0) for op in set(t['instructions_by_opcode'])|set(n['instructions_by_opcode'])}
for op,count in sorted(delta.items(),key=lambda x:abs(x[1]),reverse=True)[:12]:lines.append(f'{op:25} {count:+.0f}')
lines+=['', 'Region counts baseline/named/toggle (boundaries:first/lastIMMA and loop backedge; motion across boundaries is possible):']
for zone in b['instructions_by_region']:lines.append(zone+': '+ '/'.join(f"{r['instructions_by_region'][zone]:.0f}" for r in [b,n,t]))
lines+=['', 'PC samples:']
for name,r in result.items():
 lines.append(name+f" L2B/usefulMMA={r['l2_bytes_per_useful_mma']:.6f}, math samples onIMMA={r['math_samples_on_IMMA_pct']:.3f}%")
 for reason in ['stall_barrier','stall_short_sb','stall_long_sb','stall_math']:lines.append(reason+': '+str(r['pc_samples'][reason]['top'][:2]))
lines+=['', 'Interpretation:',
'1. Unlike the previous named overlap, toggle beats the production baseline in measured IMMA/elapsedSMcycle and same-metric Tensor utilization.',
'2. Same executed MMA count;L2 bytes/usefulMMA stays37.758..;shared69632B;0spill;1CTA/SM;baseline/toggle168registers. No occupancy or work-count shortcut.',
'3. Barrier stall rises compared to previous named, yet performance improves. Short scoreboard falls and instruction count changes; no single stall percentage explains total time.',
'4. Warp stall samples are not wall-time shares. Changes in compiler scheduling and register allocation accompany the control simplification, so causality cannot be assigned solely to barrier-ID arithmetic.',
'5. Gain is not merely higher clock: cycle-normalized useful IMMA throughput improves. Kernel-only Nsight time differs from unprofiled event and full-grid timings.',
'6. Previous two event runs improved kernel time1.371%/1.278%;full offline grid60samples/variant improved307.555->304.140ms,228.801->231.370TH/s(+1.123%).',
'7. This is a small supported gain, not evidence that the remaining gap to PeakMiner is solved.',
'Next bounded work: inspect remaining recurring operand-address/control instructions against baseline while preserving the validated handoff; retain this binary as the experimental checkpoint.',
'No production replacement/live mining in this analysis;previous correctness and2025.2.1 memcheck/racecheck/synccheck passed.']
(out/'interpretation.json').write_text(json.dumps(result,indent=2)+'\n');(root/'artifacts/handoff-lean-profile-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:28]))
