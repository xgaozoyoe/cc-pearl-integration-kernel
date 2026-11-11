"""Matched rename and wait-placement comparisons; no pool-rate claims."""
from pathlib import Path
import argparse,json,statistics as st,collections
import numpy as np
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);args=p.parse_args();out=Path(__file__).resolve().parents[1]/'artifacts/address-controls';names=list(json.loads((out/'manifest.json').read_text())['variants'])
lines=['Fastest-native fixed-SASS address-lifetime controls; all arithmetic correct.', 'Names: native; ld_rename keepswait1110; ld_release moveswait1110->1150;', 'cp_rename keepswaitf60; cp_release moveswaitf60->f70; both_release combines.', 'Same instruction addresses/count,MMA,LDSM,copies,shared memory,register allocation,barrier phases.', 'R154 reused only while dead; cp address temp ends1010; LDSM address temp starts1110 ends14f0.', 'No instruction deleted, no memory completion or stage synchronization removed.', 'Copy experiment exposes only one additional instruction before next required wait; not the cost of all copy dependencies.', 'The LDSM experiment shifts address-read completion only; does not remove operand-result waiting.', 'Rejected initial R166 build triggeredCUDA715 before benchmarking; preserved in rejected-r166. Not included in any performance claim.', '']
for filename in ('events.json','grids.json'):
 if not (out/filename).exists():continue
 d=json.loads((out/filename).read_text());lines.append(filename+': name n mean_ms median_ms p95_ms');means={}
 for name in names:
  vals=[x for r in d['measurements'] if r['name']==name for x in r['samples_ms']]
  if not vals:continue
  means[name]=st.mean(vals);lines.append(f'{name} {len(vals)} {means[name]:.6f} {np.median(vals):.6f} {np.percentile(vals,95):.6f}')
 for a,b in (('ld_rename','ld_release'),('cp_rename','cp_release'),('native','both_release')):
  if a in means and b in means:lines.append(f'{a}->{b}:time change {100*(means[b]/means[a]-1):+.4f}%,work-rate change {100*(means[a]/means[b]-1):+.4f}%')
  aa={r['repeat']:r['mean_ms'] for r in d['measurements'] if r['name']==a};bb={r['repeat']:r['mean_ms'] for r in d['measurements'] if r['name']==b}
  paired=[100*(aa[i]/bb[i]-1) for i in sorted(aa.keys() & bb.keys())]
  if paired:lines.append('Paired block work-rate changes%: '+', '.join(f'{x:+.3f}' for x in paired))
 lines.append('Unfixed clocks. Region CUDA events and full-grid wall time are different workloads; do not infer pool TH/s from region rates.')
if args.capture:
 cap=args.capture;issue=json.loads((cap/'imma-issue.json').read_text());pc=json.loads((cap/'pc-stalls.json').read_text())
 for i,name in enumerate(names):
  lines.append(name+' cycle-normalized '+json.dumps(issue[str(i)]['means']))
  target=collections.defaultdict(lambda:collections.defaultdict(float))
  for r in pc[i]['pcs']:
   address=int(r['pc_offset'],16)
   if address in (0xf60,0xf70,0x1110,0x1150):
    for key,value in r['samples'].items():target[hex(address)][key]+=value
  lines.append('All-launch targetPC samples '+json.dumps(target))
 (cap/'address-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
