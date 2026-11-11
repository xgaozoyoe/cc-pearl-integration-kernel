"""Nonzero B-input cut, separate oracle. Do not equate altered arithmetic with mining."""
from pathlib import Path
import json,argparse
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);a=p.parse_args();out=Path(__file__).resolve().parents[1]/'artifacts/cta-handoff';names=['native','cta_sync']
lines=['Correct-arithmetic candidate: 16d0 BAR.ARV->NOP;2a00 named BAR.SYNC->full CTA BAR.SYNC0.', 'No code motion or recompilation. cp.async wait, LDSM waits, MMA, transcript, loop, initial/final barriers unchanged.', 'Original own/peer ID arithmetic retained for controlled comparison; no new instructions.', 'All threads complete async copies and old-stage reads before full CTA barrier releases; next iteration may reuse shared memory safely.', 'Replacing two group barriers with a single CTA barrier is a stronger rendezvous; performance benefit is not assumed.', 'UNLOCKED region timings need fixed-clock comparison. Production library unchanged.', '']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());lines.append('UNLOCKED region events:200samples/variant,16launches/sample,10rotating blocks. ms,notfullgrid.')
 for n in names:
  x=e['summaries'][n];lines.append(n+' '+json.dumps(x))
  clocks=[int(r['telemetry_end'].split()[0]) for r in e['measurements'] if r['name']==n];lines.append(f'block-end SMclock min/max {min(clocks)}/{max(clocks)}MHz;not timeaveraged.')
 for left,right in [('native','cta_sync')]:
  x=e['summaries'];lines.append(f'{left}->{right}:time {100*(x[right]["mean_ms"]/x[left]["mean_ms"]-1):+.3f}%')
if a.capture:
 issue=json.loads((a.capture/'imma-issue.json').read_text());s=json.loads((a.capture/'summary.json').read_text())
 for i,n in enumerate(names):
  assert all(r['count_matches_sass'] for r in issue[str(i)]['launches']);lines.append(n+' fixed-clock '+json.dumps(issue[str(i)]['means']));lines.append(json.dumps({k:v['mean'] for k,v in s[str(i)]['metrics'].items()}))
 (a.capture/'cta-handoff-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
