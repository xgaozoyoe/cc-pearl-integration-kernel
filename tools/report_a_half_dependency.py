"""Nonzero B-input cut, separate oracle. Do not equate altered arithmetic with mining."""
from pathlib import Path
import json,argparse
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);a=p.parse_args();out=Path(__file__).resolve().parents[1]/'artifacts/a-half-dependency';names=['native','b_as_a_half','b_as_a_half_release']
lines=['Fixed-SASS A dependency diagnostic; incorrect Pearl arithmetic except native.', 'Half0 MMA uses the current nonzero B quartet for A; half1 keeps genuine A.', 'All original A/B loads, 64 static MMA, C accumulation, registers and barriers retained.', 'release moves A wait SB1 1130->11a0 and SB3 1140->11b0.', 'SB3 covers both A136 and A96: wait remains before first real use of either.', 'Only partial A dependency is removed. Source oracle independently verifies altered arithmetic.', 'Input correlation/register bank access also changes; matched wait-only comparison is the strongest control.', 'UNLOCKED times are region timings, not full grids or usable mining throughput.', '']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());lines.append('UNLOCKED region events:200samples/variant,16launches/sample,10rotating blocks. ms,notfullgrid.')
 for n in names:
  x=e['summaries'][n];lines.append(n+' '+json.dumps(x))
  clocks=[int(r['telemetry_end'].split()[0]) for r in e['measurements'] if r['name']==n];lines.append(f'block-end SMclock min/max {min(clocks)}/{max(clocks)}MHz;not timeaveraged.')
 for left,right in [('native','b_as_a_half'),('b_as_a_half','b_as_a_half_release')]:
  x=e['summaries'];lines.append(f'{left}->{right}:time {100*(x[right]["mean_ms"]/x[left]["mean_ms"]-1):+.3f}%')
if a.capture:
 issue=json.loads((a.capture/'imma-issue.json').read_text());s=json.loads((a.capture/'summary.json').read_text())
 for i,n in enumerate(names):
  assert all(r['count_matches_sass'] for r in issue[str(i)]['launches']);lines.append(n+' fixed-clock '+json.dumps(issue[str(i)]['means']));lines.append(json.dumps({k:v['mean'] for k,v in s[str(i)]['metrics'].items()}))
 (a.capture/'a-half-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
