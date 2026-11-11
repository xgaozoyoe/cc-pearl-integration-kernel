"""Nonzero B-input cut, separate oracle. Do not equate altered arithmetic with mining."""
from pathlib import Path
import json,argparse
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);a=p.parse_args();out=Path(__file__).resolve().parents[1]/'artifacts/c-dependency';names=['native','a_as_c']
lines=['Fixed-SASS nonzero C dependency diagnostic; incorrect Pearl arithmetic except native.', 'All 64 IMMA C inputs use the current A quartet as INT32 bit patterns.', 'Original A/B loads, MMA destinations, loop, transcript instructions, scheduling controls, registers and barriers retained.', 'No compiler runs on timed binary; independent source oracle checks altered arithmetic.', 'C recurrence removed, but destination WAW hazards and transcript consumers remain.', 'A/C register aliasing and numeric switching activity change too; this is not an implementable mining optimization.', 'UNLOCKED events are region timings, not full grids. Fixed-clock counters required.', '']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());lines.append('UNLOCKED region events:200samples/variant,16launches/sample,10rotating blocks. ms,notfullgrid.')
 for n in names:
  x=e['summaries'][n];lines.append(n+' '+json.dumps(x))
  clocks=[int(r['telemetry_end'].split()[0]) for r in e['measurements'] if r['name']==n];lines.append(f'block-end SMclock min/max {min(clocks)}/{max(clocks)}MHz;not timeaveraged.')
 for left,right in [('native','a_as_c')]:
  x=e['summaries'];lines.append(f'{left}->{right}:time {100*(x[right]["mean_ms"]/x[left]["mean_ms"]-1):+.3f}%')
if a.capture:
 issue=json.loads((a.capture/'imma-issue.json').read_text());s=json.loads((a.capture/'summary.json').read_text())
 for i,n in enumerate(names):
  assert all(r['count_matches_sass'] for r in issue[str(i)]['launches']);lines.append(n+' fixed-clock '+json.dumps(issue[str(i)]['means']));lines.append(json.dumps({k:v['mean'] for k,v in s[str(i)]['metrics'].items()}))
 (a.capture/'c-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
