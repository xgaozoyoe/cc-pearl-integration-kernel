"""Report true delayed named-sync experiment, not deleted synchronization."""
from pathlib import Path
import json,argparse
root=Path(__file__).resolve().parents[1];out=root/'artifacts/late-named-sync'
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);args=p.parse_args()
lines=['Frozen baseline, GPU0 only. Production libraries unchanged.',
'Matched control: named sync + independent P6 instruction + oldID toggle ->33instruction address prefix.',
'Candidate:33instruction address prefix -> named sync + independent P6 instruction + oldID toggle.',
'Both preserve early BAR.ARV,64late sync arrivals,initial/final CTA barriers,all cp.wait and shared-memory accesses.',
'Old R161 is unchanged by prefix, consumed before toggle. P6 dummy retains proven scheduling gap without clobbering prepared P2.',
'Prefix begins with original waitmask6 protecting operand reads of previous transcript stores/early arrival.',
'Final K iteration has original sync/exit path; first iteration executes original prefix; all other prefix executions branch directly to dc0.',
'Same registers168,dynamicshared69632,local0,384threads,1CTA/SM.64staticIMMA and16LDSM unchanged.',
'No new CTA-wide synchronization and no deleted synchronization. Same instruction multiset between matched candidates.',
'Code movement changes layout and dependencies; causal interpretation still needs measured timing/counters.',
'']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());s=e['summaries'];b=s['native']['mean_ms'];c=s['matched_control']['mean_ms']
 lines.append('Unlocked region8192x8192,K4096,R128.200samples/variant,16launches/sample,10rotating blocks.')
 for n,v in s.items():lines.append(n+' '+json.dumps(v)+f' time_change_vs_native_pct={100*(v["mean_ms"]/b-1):+.6f} time_change_vs_control_pct={100*(v["mean_ms"]/c-1):+.6f}')
if (out/'grids.json').exists():
 import numpy as np
 e=json.loads((out/'grids.json').read_text());lines.append('Offline full-grid wall time;2^46equivalent hashes/grid,not pool rate.')
 for name in ['native','late_sync']:
  vals=[v for r in e['measurements'] if r['name']==name for v in r['samples_ms']]
  if vals:lines.append(name+' '+json.dumps(dict(n=len(vals),mean_ms=float(np.mean(vals)),median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),equivalent_ths=2**46/(np.mean(vals)/1000)/1e12)))
if (out/'racecheck.log').exists():lines.extend(x for x in (out/'racecheck.log').read_text().splitlines() if 'SUMMARY' in x)
if (out/'pow-validation.json').exists():
 rows=json.loads((out/'pow-validation.json').read_text());assert all(r['transcript_pass'] and r['digest_pass'] for r in rows);lines.append(f'Full transcript+PoW:{len(rows)}cases PASS.')
if args.capture:
 i=json.loads((args.capture/'imma-issue.json').read_text());b=i['0']['means']['imma_per_SM_elapsed_cycle']
 for k,n in enumerate(['native','matched_control','late_sync']):
  row=i[str(k)];assert all(r['count_matches_sass'] for r in row['launches']);lines.append(n+' base-clock-policy '+json.dumps(row['means'])+f' IMMA_per_cycle_change_pct={100*(row["means"]["imma_per_SM_elapsed_cycle"]/b-1):+.6f}')
 (args.capture/'late-named-sync-report.txt').write_text('\n'.join(lines)+'\n')
(out/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
