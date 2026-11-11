"""Nonzero B-input cut, separate oracle. Do not equate altered arithmetic with mining."""
from pathlib import Path
import json,argparse
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);a=p.parse_args();out=Path(__file__).resolve().parents[1]/'artifacts/nonzero-dependency';names=['native','a_as_b','a_as_b_release']
lines=['Fastest-native fixed-SASS nonzero diagnostic. Incorrect Pearl arithmetic except native.', 'B input uses currentlyrequired A fragment:half0=Aregisters0,1;half1=Aregisters2,3.', 'B data is derived from real random matrix data, not constantzero. No new instructions/registers/initialization.', 'Original Bglobalcopy/LDSM traffic and all64MMA sites preserved; Caccumulation remains real and nonzero.', 'Only a_as_b_release moves3Bwaits after independent IMMA; completion still enforced before reuse.', 'Source oracle compiled independently with equivalent A-derived Bsemantics; oracle is NOT timed.', 'This removes additional Bload dependency conditional on Aready. It also changes operand correlation/register-file reads.', 'Does not model a real prefetchedB optimization. Fixed clock counters still needed to isolate DVFS.', '']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());lines.append('UNLOCKED region events:200samples/variant,16launches/sample,10rotating blocks. ms,notfullgrid.')
 for n in names:
  x=e['summaries'][n];lines.append(n+' '+json.dumps(x))
  clocks=[int(r['telemetry_end'].split()[0]) for r in e['measurements'] if r['name']==n];lines.append(f'block-end SMclock min/max {min(clocks)}/{max(clocks)}MHz;not timeaveraged.')
 for left,right in [('native','a_as_b'),('a_as_b','a_as_b_release')]:
  x=e['summaries'];lines.append(f'{left}->{right}:time {100*(x[right]["mean_ms"]/x[left]["mean_ms"]-1):+.3f}%')
if a.capture:
 issue=json.loads((a.capture/'imma-issue.json').read_text());s=json.loads((a.capture/'summary.json').read_text())
 for i,n in enumerate(names):
  assert all(r['count_matches_sass'] for r in issue[str(i)]['launches']);lines.append(n+' fixed-clock '+json.dumps(issue[str(i)]['means']));lines.append(json.dumps({k:v['mean'] for k,v in s[str(i)]['metrics'].items()}))
 (a.capture/'nonzero-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
