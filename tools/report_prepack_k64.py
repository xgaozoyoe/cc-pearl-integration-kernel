"""Measured prepack costs, comparable controls, and explicit limits."""
from pathlib import Path
import json,re,statistics as st
import numpy as np
root=Path(__file__).resolve().parents[1];p=root/'artifacts/prepack-k64';e=json.loads((p/'events.json').read_text());g=json.loads((p/'grids.json').read_text());pack=json.loads((p/'pack-events.json').read_text())
lines=['GPU0 only; offline K4096/rank128; no production deployment.', 'Layout: [row-block][K64-stage][shared-swizzled bytes]. Ablock256,Bblock192.', 'CTA256x192,384threads,warp64x64,2stages,64IMMA+16LDSM per K64 unchanged.', 'Native is frozen fastest fixed-SASS baseline; source and packed are recompiled controls.', 'Both recompiled controls lose native early-publication scheduling; performance vs source is not proof of additive gains over native.', '', 'Kernel events8192x8192:200 samples/variant,16 launches/sample,10 rotating blocks.', 'variant mean_ms median_ms p95_ms registers local_bytes shared_bytes CTAs/SM']
for n,x in e['summaries'].items():
 r=e['resources'][n];lines.append(f"{n} {x['mean_ms']:.6f} {x['median_ms']:.6f} {x['p95_ms']:.6f} {r[0]} {r[1]} 69632 {r[3]}")
lines+=['',f"Pack A8192+B8192 pair:mean {pack['mean_ms']:.6f}ms,70 CUDA event samples after10 warmup.",f"Amortized model16 pairs/grid: {16*pack['mean_ms']:.6f}ms. This is an estimate; full-grid wall times below include actual pack launches.", 'Full grid:noised A regions packed once per row region; B regions packed once per grid when generated.', 'Original A/B and Merkle/proof/target paths unchanged. Pack kernels on default stream follow noise and precede GEMM.', 'Steady-state extra retained buffers:32MiB A +16*32.25MiB B =548MiB.', 'Raw bytes read per grid1GiB; packed bytes written1028MiB (B region padding included).', '', 'Full-grid wall:60 samples/variant,6 balanced rotating blocks,12grids/block discardfirst2.', 'Includes actual setup/noise/V3/Merkle/pack/GEMM/transcript/PoW. Target0:no network or proof delivery.', 'Clock/power settings untouched and unlocked. TH/s is consensus-equivalent work, not pool hashrate.', 'variant mean_ms median_ms p95_ms grids_per_sec equivalent_THs']
means={};batches={}
for n in e['summaries']:
 rs=[r for r in g['measurements'] if r['name']==n];v=[x for r in rs for x in r['samples_ms']];avg=st.mean(v);means[n]=avg;batches[n]={r['repeat']:r['mean_ms'] for r in rs}
 lines.append(f'{n} {avg:.3f} {np.median(v):.3f} {np.percentile(v,95):.3f} {1000/avg:.6f} {2**46/1e9/avg:.3f}')
for control in ('source','native'):
 lines.append(f"Packed vs {control}:work-rate change {100*(means[control]/means['packed']-1):+.3f}%; time change {100*(means['packed']/means[control]-1):+.3f}%")
 lines.append('Paired block work-rate changes%: '+', '.join(f'{100*(batches[control][i]/batches["packed"][i]-1):+.3f}' for i in sorted(batches[control])))
lines+=['', 'Static disassembly counts (not dynamic instruction throughput):']
for n in ('source','packed'):
 s=(p/(n+'.sass')).read_text();rows=re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)
 lines.append(n+' '+str({op:sum(op in x for _,x in rows) for op in ('IMMA.','LDSM.','LDGSTS','BAR.','REDUX','LDL','STL')}))
lines+=['', 'Important limits: original warp copies already cover full32B sectors. No measured L2-byte reduction or Tensor utilization claim.', 'Prepack changes address/control instructions and register allocation as well as access continuity; cannot isolate their contributions.', 'Observed regional pack latency is not a bandwidth ceiling or guaranteed cost inside the miner.', 'Next bounded experiment if continuing: restore comparable safe early publication in both source and packed, then retest full-grid including packing.', 'No new production baseline selected by this experiment.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
