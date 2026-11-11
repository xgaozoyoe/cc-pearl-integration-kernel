"""Copy-completion wait x named-handoff ablation. Offline diagnostic only."""
from pathlib import Path
import json,argparse
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);args=p.parse_args();out=Path(__file__).resolve().parents[1]/'artifacts/wait-vs-handoff';names=['native','no_copy_wait','no_barrier','both']
lines=['Offline wrong-arithmetic 2x2 diagnostic; exact frozen binary, no code motion/recompilation.', 'Factor1 replaces16c0 DEPBAR.LE SB0,0 (main-loop cp.async.wait_group0) withNOP.', 'Factor2 replaces16d0 BAR.ARV and2a00 BAR.SYNC withNOP.', 'All async copies/commits, LDSM, IMMA, prologue wait/barrier, final CTA barrier, transcript and loop retained.', 'Registers168, shared69632,384threads,1CTA/SM unchanged. Static stalls/wait masks retained; removed BAR producer annotations cleared.', 'Removing copy completion does not remove implicit queue throttling or all address/register waits.', 'Wrong data/races can change numerical switching and warp phasing; these timings are not safe optimization gains.', 'No per-warp arrival timestamps collected. This compares the two synchronization mechanisms and their interaction.', '']
def compare(t,label):
 lines.append(label+': '+json.dumps(t))
 lines.append(f'Copy-wait removal gain with barriers {100*(1-t["no_copy_wait"]/t["native"]):.4f}%; without barriers {100*(1-t["both"]/t["no_barrier"]):.4f}%.')
 lines.append(f'Barrier removal gain original copy wait {100*(1-t["no_barrier"]/t["native"]):.4f}%; removed copy wait {100*(1-t["both"]/t["no_copy_wait"]):.4f}%.')
 interaction=t['native']-t['no_copy_wait']-t['no_barrier']+t['both']
 lines.append(f'Interaction (T00-T10)-(T01-T11):{interaction:.9f} {label}; {100*interaction/t["native"]:.4f}% of native. Positive means part of copy-wait-removal benefit overlaps barrier-removal benefit.')
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());lines.append('Unlocked events:200samples/variant,16launches/sample,10rotating blocks. Region8192x8192x4096, not full mining grid.')
 for n in names:lines.append(n+' '+json.dumps(e['summaries'][n]))
 compare({n:e['summaries'][n]['mean_ms'] for n in names},'unlocked ms')
if args.capture:
 issues=json.loads((args.capture/'imma-issue.json').read_text());s=json.loads((args.capture/'summary.json').read_text())
 for i,n in enumerate(names):
  assert all(x['count_matches_sass'] for x in issues[str(i)]['launches'])
  lines.append(n+' '+json.dumps(issues[str(i)]['means']));lines.append(json.dumps({k:v['mean'] for k,v in s[str(i)]['metrics'].items()}))
 compare({n:issues[str(i)]['means']['kernel_ms'] for i,n in enumerate(names)},'base-clock ms')
 compare({n:1/issues[str(i)]['means']['imma_per_SM_elapsed_cycle'] for i,n in enumerate(names)},'SM elapsed cycles per IMMA')
 (args.capture/'wait-vs-handoff-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
