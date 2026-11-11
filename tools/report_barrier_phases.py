"""Compare K-parity handoffs independently; not individual opcode latency."""
from pathlib import Path
import argparse,json
root=Path(__file__).resolve().parents[1];out=root/'artifacts/barrier-phases'
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);a=p.parse_args()
names=['native','predicate_control','no_plain_handoff','no_fold_handoff']
lines=['Frozen baseline, GPU0 only. Offline deliberately racy diagnostics; never mine.',
'Control:10a0 ISETP second discarded PT result ->P6, preserving P2; P6=!fold and unused elsewhere.',
'no_plain_handoff:both16d0 arrive and2a00 sync execute only in fold rounds; IDs4/5 skipped.',
'no_fold_handoff:both16d0 arrive and2a00 sync execute only in non-fold rounds; IDs6/7 skipped.',
'Each remaining named barrier still has384participants. Initial/final barriers,cp.wait,LDSM guards retained.',
'All64static IMMA,16LDSM,instruction positions,control fields,launch geometry and register allocation retained.',
'Each deletion skips32 paired handoffs per warp. Fold/no-fold computations themselves are unchanged.',
'This isolates phase-dependent rendezvous effects, NOT independent BAR.ARV/BAR.SYNC opcode costs.',
'Skipped synchronization can change values,switching activity and following-round phasing. Effects are not additive.',
'']
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());s=e['summaries'];b=s['predicate_control']['mean_ms']
 lines.append('Unlocked CUDA events,8192x8192 region,K4096.200samples/variant,16launches/sample,rotating blocks.')
 for n in names:lines.append(n+' '+json.dumps(s[n])+f' vs_control_time_reduction_pct={100*(1-s[n]["mean_ms"]/b):+.4f}')
 lines.append(f'Predicate control vs native time change={100*(b/s["native"]["mean_ms"]-1):+.4f}%')
if a.capture:
 i=json.loads((a.capture/'imma-issue.json').read_text());b=i['1']['means']['imma_per_SM_elapsed_cycle']
 for k,n in enumerate(names):
  x=i[str(k)];assert all(r['count_matches_sass'] for r in x['launches'])
  lines.append(n+' base-clock-policy '+json.dumps(x['means'])+f' vs_control_IMMA_per_cycle_gain_pct={100*(x["means"]["imma_per_SM_elapsed_cycle"]/b-1):+.6f}')
 (a.capture/'barrier-phases-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
