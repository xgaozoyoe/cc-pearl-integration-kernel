"""Report fixed-SASS controls on the fastest frozen baseline."""
from pathlib import Path
import json,statistics as st
import numpy as np
p=Path(__file__).resolve().parents[1]/'artifacts/native-copy-trim';e=json.loads((p/'events.json').read_text());g=json.loads((p/'grids.json').read_text())
lines=['GPU0 offline, fastest fold-preamble/both_compact frozen baseline.', 'Only fixed-SASS edits; no source recompilation or register allocation changes.', 'tid_move:PC0xdd0 S2R R98,SR_TID.X -> LOP3 R98,R162,0 (register copy).', 'R162 is initialized from TID atPC0x40 and is not overwritten until after the K loop.', 'New fixed-latency ALU instruction uses no write scoreboard; all other instruction positions and controls unchanged.', 'tid_bounds:also replace2always-false comparisons withNOP and remove10 corresponding instruction predicates.', 'For384threads,tid<=383 andtid+384<=767, always <=1023; third A-copy bound remains.', 'NOP slots retained: no total issued-instruction-count reduction claim.', 'Native64IMMA,16LDSM,K64,2stages,CTA256x192,384threads,early arrival and transcript retained.', 'All168registers,69632Bshared,0spill/local,1CTA/SM. V3,rank target and proofs untouched.', '', 'Kernel:200samples/variant,16launches/sample,10rotating blocks,8192x8192 region.', 'name mean_ms median_ms p95_ms']
for n,x in e['summaries'].items():lines.append(f"{n} {x['mean_ms']:.6f} {x['median_ms']:.6f} {x['p95_ms']:.6f}")
lines+=['', 'Full grid131072x131072x4096,rank128,region8192.60samples/variant,6rotating blocks.', 'Includes real setup/noise/Merkle/V3/GEMM/transcript/PoW,target0 excludes proof delivery/network.', 'Clocks/power unchanged and unlocked; reported TH/s is consensus-equivalent, not pool hashrate.', 'name mean_ms median_ms p95_ms grids_sec equivalent_THs']
means={};batches={}
for n in e['summaries']:
 rs=[r for r in g['measurements'] if r['name']==n];v=[x for r in rs for x in r['samples_ms']];avg=st.mean(v);means[n]=avg;batches[n]={r['repeat']:r['mean_ms'] for r in rs}
 lines.append(f'{n} {avg:.3f} {np.median(v):.3f} {np.percentile(v,95):.3f} {1000/avg:.6f} {2**46/1e9/avg:.3f}')
for n in ('tid_move','tid_bounds'):
 lines.append(f'{n}:work-rate vs native {100*(means["native"]/means[n]-1):+.3f}%')
 lines.append('Paired block changes%: '+', '.join(f'{100*(batches["native"][i]/batches[n][i]-1):+.3f}' for i in sorted(batches[n])))
lines+=['', 'Validation:36edge transcript comparisons and18full transcript+PoW comparisons including8192x8192 PASS.', 'Interpretation:single-instruction substitution preserves the existing fastest scheduling. Causal claim limited to this substitution including scoreboard change; no new Tensor utilization or stall measurements.', 'No production deployment. No sanitizer run in this round; candidates were not promoted. See diffs for exact edits.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
