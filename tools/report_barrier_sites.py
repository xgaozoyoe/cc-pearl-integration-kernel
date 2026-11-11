"""Report individual prologue/epilogue barrier ablations; timings are not additive."""
from pathlib import Path
import json,argparse
root=Path(__file__).resolve().parents[1];out=root/'artifacts/barrier-sites'
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);a=p.parse_args()
names=['native','no_initial','no_final'];lines=[
'Frozen baseline single-site ablations, GPU0 only, offline and deliberately racy.',
'no_initial:0xba0 BAR.SYNC0 ->NOP; own-thread cp.async wait retained.',
'no_final:0x2a50 BAR.SYNC0 ->NOP; output instructions retained.',
'All main-loop named arrive/sync,IMMA,LDSM,copies,launch geometry and168regs unchanged.',
'Positive gain=(native_time-variant_time)/native_time. Effects are configuration-specific, not additive.',
'No correctness claim. Removing synchronization can change operand values, warp phasing and switching activity.',
'No isolated main-loop BAR.ARV or BAR.SYNC cost measured in this experiment.',
'']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());s=e['summaries'];b=s['native']['mean_ms']
 lines.append('Unlocked CUDA event region8192x8192,K4096; not full-grid time.')
 for n in names:lines.append(n+' '+json.dumps(s[n])+f' mean_delta_us={(b-s[n]["mean_ms"])*1000:+.6f}, time_reduction_pct={100*(1-s[n]["mean_ms"]/b):+.4f}')
 lines.append('Small unlocked differences need fixed-clock/cycle-normalized confirmation before attribution.')
if a.capture:
 issues=json.loads((a.capture/'imma-issue.json').read_text());b=issues['0']['means']
 for i,n in enumerate(names):
  assert all(x['count_matches_sass'] for x in issues[str(i)]['launches'])
  x=issues[str(i)]['means'];lines.append(n+' base-clock-policy '+json.dumps(x)+f' IMMA_per_cycle_gain_pct={100*(x["imma_per_SM_elapsed_cycle"]/b["imma_per_SM_elapsed_cycle"]-1):+.6f}')
 (a.capture/'barrier-sites-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
