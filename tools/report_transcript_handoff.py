"""Report event and full-grid evidence without deploying the experimental backend."""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parents[1];p=root/'artifacts'
e=json.loads((p/'transcript-handoff-events.json').read_text());g=json.loads((p/'transcript-handoff-results.json').read_text());names=['baseline','late','overlap'];result={}
lines=['Correct-arithmetic transcript / split-barrier handoff experiment (GPU0)','', 'variant   kernel_mean_ms kernel_median_ms kernel_p95_ms grid_mean_ms grid_median_ms grid_p95_ms grids/s effective_TH/s']
for v,name in enumerate(names):
 ev=np.array([x for r in e['measurements'] if r['variant']==v for x in r['samples_ms']]);gv=np.array([x for r in g['measurements'] if r['variant']==v for x in r['samples_ms']]);assert len(ev)==240 and len(gv)==60
 r=dict(kernel_samples=len(ev),grid_samples=len(gv),kernel_mean_ms=float(ev.mean()),kernel_median_ms=float(np.median(ev)),kernel_p95_ms=float(np.percentile(ev,95)),grid_mean_ms=float(gv.mean()),grid_median_ms=float(np.median(gv)),grid_p95_ms=float(np.percentile(gv,95)),grids_sec=1000/float(gv.mean()),effective_ths=2**46/float(gv.mean())/1e9,resources=e['resources'][str(v)])
 result[name]=r;lines.append(f"{name:9} {r['kernel_mean_ms']:.6f}       {r['kernel_median_ms']:.6f}         {r['kernel_p95_ms']:.6f}      {r['grid_mean_ms']:.3f}       {r['grid_median_ms']:.3f}        {r['grid_p95_ms']:.3f}     {r['grids_sec']:.4f}   {r['effective_ths']:.3f}")
b,l,o=[result[n] for n in names]
lines+=['',f"Early arrival vs late arrival: kernel time reduction {100*(1-o['kernel_mean_ms']/l['kernel_mean_ms']):.3f}%; grid time reduction {100*(1-o['grid_mean_ms']/l['grid_mean_ms']):.3f}%.",f"Early arrival vs original baseline: kernel time increase {100*(o['kernel_mean_ms']/b['kernel_mean_ms']-1):.3f}%; grid time increase {100*(o['grid_mean_ms']/b['grid_mean_ms']-1):.3f}%; effective throughput change {100*(o['effective_ths']/b['effective_ths']-1):.3f}%.", '',
'Conclusion: no deployable performance gain. Earlier registration is faster than late registration in this matched experiment, but both lose to native baseline.',
'No cycle-normalized counter result yet. Frequency and generated scheduling differences limit attribution.',
'New path includes async barrier arrival, MEMBAR.ALL.CTA, ATOMS.ARRIVE and shared-memory polling (LDS,R2UR,predicate,branch).',
'This is a candidate explanation for cost, not a measured breakdown of overhead.',
'SASS confirms early arrival before transcript REDUX and wait after transcript; source-only rearrangement is not being mistaken for actual overlap.',
'All variants: 256x192 CTA,384 threads,168 registers/thread,0 spills/local memory,1CTA/SM,64IMMA+16LDSM perK64.',
'New barrier costs8 shared bytes:69632 ->69640; occupancy unchanged.',
'Rebuilt baseline SASS instruction text and PCs match frozen accepted candidate exactly.',
'',
'24 complete transcript+PoW digest comparisons PASS, including2 full8192x8192 inputs; benchmark preflight also compares full transcript.',
'Compute Sanitizer2025.2.1: racecheck0 hazards/errors/warnings on18 small/edge comparisons; full-size memcheck0 errors.',
'Compute Sanitizer2024.3 reported races on the same binary; both logs retained. Official later release notes document related cp.async.mbarrier.arrive fixes.',
'https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html',
'No system CUDA/driver changes: verified standalone NVIDIA tool extracted under artifacts/sanitizer-12.9.79.',
'No production library/seed/target/proof changes, no live mining. GPU2 miner was not stopped.',
'',
'Synchronization design and invariants: artifacts/transcript-handoff-design.txt.',
'Next counters, if investigating this mechanism further: sudo bash tools/profile_transcript_handoff.sh 2270585',
'Compare cycle-normalized throughput, barrier vs shared-polling short scoreboard, MEMBAR/ATOMS stalls; do not blindly move arrival earlier again.']
(p/'transcript-handoff-report.json').write_text(json.dumps(result,indent=2)+'\n');(p/'transcript-handoff-report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
