from pathlib import Path
import json,numpy as np
p=Path(__file__).resolve().parents[1]/'artifacts/transcript-current-ab';names=['full','fold_sink','gemm_sink'];combined={n:[] for n in names}
lines=['Current8+4 transcript ablation: fixed-SASS, GPU0, no Nsight.',
'8192x8192 region,K4096. Kernel-only, not a complete mining grid. No PoW in any mode.',
'2 independent runs,9rotating blocks/run,180samples/mode/run,16 launches/sample.',
'Unfixed clocks. All modes168registers,69632Bshared,384threads,1CTA/SM,0local.',
'run mode mean_ms median_ms p95_ms synchronized_batch_wall_ms_per_launch']
for run in (1,2):
 d=json.loads((p/f'events-run{run}.json').read_text());s=d['summaries']
 for n in names:
  x=s[n];combined[n].extend(v for r in d['measurements'] if r['name']==n for v in r['samples_ms'])
  lines.append(f'{run} {n} {x["mean_ms"]:.6f} {x["median_ms"]:.6f} {x["p95_ms"]:.6f} {x["wall_mean_ms"]:.6f}')
 a=s['full']['mean_ms'];f=s['fold_sink']['mean_ms'];b=s['gemm_sink']['mean_ms']
 lines += [f'run{run}: net transcript+epilogue delta={a-b:.6f}ms, {100*(a-b)/a:.3f}% of full time; full/gemm={a/b:.6f}.',
 f'run{run}: fold delta={f-b:.6f}ms; output-epilogue differential={a-f:.6f}ms.']
 paired=[]
 for repeat in range(9):
  rr={r['name']:r['mean_ms'] for r in d['measurements'] if r['repeat']==repeat};paired.append(100*(rr['full']-rr['gemm_sink'])/rr['full'])
 lines.append('Paired net savings%: '+', '.join(f'{x:.3f}' for x in paired))
means={n:float(np.mean(v)) for n,v in combined.items()};a,f,b=[means[n] for n in names]
lines += ['', 'Combined mean ms: '+json.dumps(means),
 f'Net delta {a-b:.6f}ms / region, {100*(a-b)/a:.3f}% of full kernel; throughput ratio {a/b:.6f}.',
 f'Fold component {f-b:.6f}ms = {100*(f-b)/a:.3f}% of full kernel.',
 f'Epilogue differential {a-f:.6f}ms = {100*(a-f)/a:.3f}% of full kernel.',
 'Same-sink control isolates fold path; epilogue differential includes replacing output with a nonzero-cost sample sink.',
 'No compiler reallocation/DCE in main kernel. All64staticIMMA/16LDSM and copy/handshake words identical.',
 'Final accumulator dependencies preserved. Full transcript matches accepted reference;',
 'final R120 per-thread samples match between sink modes for all tested inputs (not all accumulator elements).',
 'This measures removable direct work under current schedule. It neither isolates all indirect transcript constraints',
 'nor establishes PeakMiner transcript speed. Do not interpret invalid diagnostic modes as mining throughput.',
 'Production library and current8+4 checkpoint unchanged. No dependency-breaking C variant tested.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
