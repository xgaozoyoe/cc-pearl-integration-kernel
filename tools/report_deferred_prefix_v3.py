"""Correct-arithmetic deferred-blocking experiment, including relocation overhead."""
from pathlib import Path
import json,argparse
root=Path(__file__).resolve().parents[1];out=root/'artifacts/deferred-prefix-v3'
parser=argparse.ArgumentParser();parser.add_argument('capture',nargs='?',type=Path);args=parser.parse_args()
lines=['Frozen baseline deferred-blocking experiment, GPU0 only, production unchanged.',
'Continuing path:termination check -> original named sync -> original predicate instruction -> ID toggle ->33instruction next-prefix -> branch to dc0.',
'Relocated control has an extra BRA before prefix; candidate replaces exactly that BRA with NOP.',
'Last iteration preserves original barrier/exit path. First iteration uses original prefix once.',
'Candidate removes a branch before33prefix instructions; whether blocking actually defers across all33requiresPCsampling.',
'All64sync arrivals per warp remain, paired early arrivals remain; shared accesses still follow the final branch.',
'Prefix executes64times total; IMMA64static/warp/K64,LDSM16,copies,transcript unchanged.',
'168regs,69632dynamic shared bytes,0local bytes,384threads,1CTA/SM unchanged.',
'Compare with native to include extra branch/predicate and code-layout overhead, not just relocated control.',
'Candidate/control comparison changes a branch as well as where deferred wait blocks: not pure barrier latency.',
'',
'Failure isolation:',
'Grow-only,loop-branch,bar-relocation-only,and guarded-exit-only controls passed.',
'v2 full control timed out. v3 restores original PLOP3 between BAR.SYNC and ID toggle and passes.',
'This is evidence of a machine-scheduling sensitivity; precise hardware cause not established (code layout also shifts).',
'Old P2 clobber also corrected: prepared next-round P2 goes directly to dc0, not back through2a10.',
'']
if (out/'events.json').exists():
 s=json.loads((out/'events.json').read_text())['summaries'];b=s['native']['mean_ms'];c=s['relocated_control']['mean_ms']
 lines.append('Unlocked region8192x8192,K4096.200samples/variant,16launches/sample,10rotating blocks.')
 for n,v in s.items():lines.append(n+' '+json.dumps(v)+f' time_change_vs_native_pct={100*(v["mean_ms"]/b-1):+.6f} time_change_vs_control_pct={100*(v["mean_ms"]/c-1):+.6f}')
if (out/'grids.json').exists():
 import numpy as np
 e=json.loads((out/'grids.json').read_text());lines.append('Offline full-grid wall time,2^46equivalent hashes/grid;not pool rate.')
 for name in ['native','deferred_prefix']:
  vals=[v for r in e['measurements'] if r['name']==name for v in r['samples_ms']]
  if vals:lines.append(name+' '+json.dumps(dict(n=len(vals),mean_ms=float(np.mean(vals)),median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),equivalent_ths=2**46/(np.mean(vals)/1000)/1e12)))
if (out/'racecheck.log').exists():
 log=(out/'racecheck.log').read_text();lines.extend(x for x in log.splitlines() if 'SUMMARY' in x)
if (out/'pow-validation.json').exists():
 rows=json.loads((out/'pow-validation.json').read_text());assert all(r['transcript_pass'] and r['digest_pass'] for r in rows);lines.append(f'Full transcript+PoW:{len(rows)}cases PASS.')
if args.capture:
 i=json.loads((args.capture/'imma-issue.json').read_text());base=i['0']['means']['imma_per_SM_elapsed_cycle']
 for k,n in enumerate(['native','relocated_control','deferred_prefix']):
  row=i[str(k)];assert all(x['count_matches_sass'] for x in row['launches']);lines.append(n+' fixed-clock-policy '+json.dumps(row['means'])+f' IMMA_per_cycle_change_pct={100*(row["means"]["imma_per_SM_elapsed_cycle"]/base-1):+.6f}')
 (args.capture/'deferred-prefix-v3-report.txt').write_text('\n'.join(lines)+'\n')
(out/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
