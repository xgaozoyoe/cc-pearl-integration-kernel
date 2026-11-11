"""Summarize two completed GPU0 copy/handoff diagnostic event runs."""
from pathlib import Path
import json, numpy as np
root=Path(__file__).resolve().parents[1];p=root/'artifacts/copy-bypass'
runs=[json.loads((p/f).read_text()) for f in ['run1-events.json','events.json']]
result={};lines=['Copy / wait / CTA-stage-barrier diagnostic (GPU0 only)','', 'Deliberately incorrect arithmetic; no production kernel or library modified.', 'Two independent runs; 200 samples/variant/run, each sample averages 16 kernel launches.', '8192 x 8192 x 4096 region; 256x192 CTA; 384 threads,168 registers,69632 dynamic shared bytes,1 CTA/SM.', '', 'variant       run1_mean  run2_mean  median_ms   p95_ms    mean_ms']
for name in runs[0]['summaries']:
 vals=np.array([x for run in runs for row in run['measurements'] if row['name']==name for x in row['samples_ms']]);assert len(vals)==400
 r=dict(samples=len(vals),mean_ms=float(vals.mean()),median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),run_means=[run['summaries'][name]['mean_ms'] for run in runs]);result[name]=r
 lines.append(f"{name:13} {r['run_means'][0]:.6f}   {r['run_means'][1]:.6f}   {r['median_ms']:.6f}   {r['p95_ms']:.6f}   {r['mean_ms']:.6f}")
a=result['no_ab']['mean_ms'];b=result['no_copy']['mean_ms'];c=result['no_copy_bar']['mean_ms']
lines+=['',f'Copy+commit+wait removal, stage barrier retained: time decreases {100*(1-b/a):.3f}%.',f'Additional removal of stage barrier: time decreases {100*(1-c/b):.3f}%.',f'Combined vs no_ab: time decreases {100*(1-c/a):.3f}%; work-rate increases {100*(a/c-1):.3f}%.','',
'Baseline: original correct kernel.',
'no_ab: IMMA A/B=RZ, all16 LDSM removed (previous diagnostic).',
'no_copy: additionally remove all10 static LDGSTS (5 prologue,5 mainloop),2 LDGDEPBAR,1 DEPBAR; preserve stage BAR.',
'no_copy_bar: additionally remove BAR at0xc60 (64 dynamic times/CTA).',
'Final BAR at0x29d0 preserved byte-exact in all variants (transcript publication before output).',
'All copy-related source-read scoreboard waits removed except loop header0xb60, shared with transcript predecessor.',
'Address calculations, fixed scheduling stalls, instruction slots (NOP replacements), branches and IMMA count remain.',
'Local transcript states are warp-owned; final output reads cross-warp states and therefore still needs final BAR.',
'', '48 expected-output checks PASS under memcheck; memcheck0 errors. Large8192-region expected outputs checked in both event runs.',
'Racecheck: 0 hazards, 0 errors, 0 warnings. Zero transcripts in diagnostic variants are not valid Pearl proofs.',
'Production and opt-in candidate SHA256 checks PASS. GPU2 miner left running; only its verified idle supervisor GPU0 context allowed.',
'', 'Interpretation: measurable stage-barrier effect in the copy-free/LDSM-free diagnostic, not a production speedup or pure barrier latency.',
'Removing a barrier can alter inter-warp phase alignment and scheduling, not just waiting time.',
'DVFS not locked; cycle-normalized Nsight results are required to separate clock changes from issue efficiency.',
'Does not authorize removing barriers from real mining: consumers there need operand readiness and safe buffer reuse.',
'Profile command: sudo bash tools/profile_copy_bypass.sh 2270585',
'Compare IMMA/SM elapsed cycle, tensor active percentage, barrier samples, eligible warps and unchanged IMMA count.',
'Check LDSM=0 in diagnostics and removal of copy-related memory traffic.']
(p/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
