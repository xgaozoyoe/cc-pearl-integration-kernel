"""Nonzero B-input cut, separate oracle. Do not equate altered arithmetic with mining."""
from pathlib import Path
import json,argparse
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);a=p.parse_args();out=Path(__file__).resolve().parents[1]/'artifacts/barrier-removal';names=['native','no_named_barriers']
lines=['Deliberately racy offline diagnostic: remove 16d0 BAR.ARV and2a00 BAR.SYNC.', 'Original64IMMA, A/B input, loop, copies, LDSM, cp.async wait, initial/final CTA barriers and transcript retained.', 'Removed barrier producer annotations; original instruction slots, wait masks and static stalls retained.', 'Registers168, shared69632, CTA384threads,1CTA/SM unchanged.', 'Output invalid. Timing difference includes altered warp phasing and shared-memory races; not isolated barrier instruction latency or a safe optimization.', 'UNLOCKED region timings are DVFS-confounded; fixed-clock counters needed.', '']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());lines.append('UNLOCKED region events:200samples/variant,16launches/sample,10rotating blocks. ms,notfullgrid.')
 for n in names:
  x=e['summaries'][n];lines.append(n+' '+json.dumps(x))
  clocks=[int(r['telemetry_end'].split()[0]) for r in e['measurements'] if r['name']==n];lines.append(f'block-end SMclock min/max {min(clocks)}/{max(clocks)}MHz;not timeaveraged.')
 for left,right in [('native','no_named_barriers')]:
  x=e['summaries'];lines.append(f'{left}->{right}:time {100*(x[right]["mean_ms"]/x[left]["mean_ms"]-1):+.3f}%')
if a.capture:
 issue=json.loads((a.capture/'imma-issue.json').read_text());s=json.loads((a.capture/'summary.json').read_text())
 for i,n in enumerate(names):
  assert all(r['count_matches_sass'] for r in issue[str(i)]['launches']);lines.append(n+' fixed-clock '+json.dumps(issue[str(i)]['means']));lines.append(json.dumps({k:v['mean'] for k,v in s[str(i)]['metrics'].items()}))
 (a.capture/'barrier-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
