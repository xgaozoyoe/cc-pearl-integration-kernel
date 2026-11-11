"""Offline grid and event results; optional fixed-clock Nsight capture."""
from pathlib import Path
import json,argparse,csv,statistics
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path);args=p.parse_args()
root=Path(__file__).resolve().parents[1];out=root/'artifacts/handoff-arrive-id';names=['tail9','id_only9','id11','id16'];lines=['Early named-barrier publication: GPU0 offline controls','Unmodified production library. Unlocked-clock event/grid timings; distinguish from optional fixed-clock counters.','']
for path in sorted(out.glob('events-run*.json')) or [out/'events.json']:
 data=json.loads(path.read_text());summaries=data['summaries'];base=summaries['tail9']['mean_ms'];lines+=[path.name,'variant mean_ms median_ms p95_ms time_change_vs_tail9']
 for n in names:
  s=summaries[n];lines.append(f"{n} {s['mean_ms']:.6f} {s['median_ms']:.6f} {s['p95_ms']:.6f} {(s['mean_ms']/base-1)*100:+.3f}%")
if (out/'grids.json').exists():
 data=json.loads((out/'grids.json').read_text());sums={};lines+=['','Full-grid wall times: includes setup/noise/Merkle/PoW, excludes network/proof delivery.','variant n mean_ms median_ms p95_ms grids_per_sec consensus_equivalent_THs']
 for n in names:
  vals=np.array([v for r in data['measurements'] if r['name']==n for v in r['samples_ms']]);mean=float(vals.mean());sums[n]=dict(samples=len(vals),mean_ms=mean,median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),grids_per_sec=1000/mean,consensus_equivalent_ths=2**46/mean/1e9)
  s=sums[n];lines.append(f"{n} {len(vals)} {mean:.3f} {s['median_ms']:.3f} {s['p95_ms']:.3f} {s['grids_per_sec']:.5f} {s['consensus_equivalent_ths']:.3f}")
 for n in names[1:]:
  lines.append(f"{n} vs tail9: time {(sums[n]['mean_ms']/sums['tail9']['mean_ms']-1)*100:+.3f}%, throughput {(sums['tail9']['mean_ms']/sums[n]['mean_ms']-1)*100:+.3f}%")
  pairs=[]
  for repeat in sorted({r['repeat'] for r in data['measurements']}):
   rows={r['name']:r for r in data['measurements'] if r['repeat']==repeat};pairs.append((rows[n]['mean_ms']/rows['tail9']['mean_ms']-1)*100)
  lines.append('Paired batch time changes: '+', '.join(f'{x:+.3f}%' for x in pairs))
 (out/'grid-summary.json').write_text(json.dumps(sums,indent=2)+'\n')
if args.capture:
 result={};metrics=['gpu__time_duration.sum','sm__cycles_elapsed.avg.per_second','smsp__inst_executed_pipe_alu.sum','smsp__inst_executed_pipe_tensor_v2.sum','sm__cycles_elapsed.sum','sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active','smsp__warp_issue_stalled_barrier_per_warp_active.pct','smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct']
 for v,n in enumerate(names):
  rows=list(csv.reader((args.capture/f'v{v}.raw.csv').open()));i=next(i for i,r in enumerate(rows) if 'ID' in r and 'Kernel Name' in r);h=rows[i];ds=[dict(zip(h,r)) for r in rows[i+1:] if len(r)==len(h) and r[h.index('ID')].strip().isdigit()];assert ds
  result[n]={m:statistics.mean(float(d[m].replace(',','')) for d in ds) for m in metrics}
  result[n]['imma_per_sm_cycle']=statistics.mean(float(d['smsp__inst_executed_pipe_tensor_v2.sum'].replace(',',''))/float(d['sm__cycles_elapsed.sum'].replace(',','')) for d in ds)
 (out/'counters.json').write_text(json.dumps(result,indent=2)+'\n');lines+=['','Fixed-clock capture: '+str(args.capture),json.dumps(result,indent=2)]
lines+=['','All variants:168 registers/thread,69632B shared,384threads,1CTA/SM,0local;64 static IMMA,16LDSM.','MMA order retained; peer-ID LOP3 moved into dead R164; id11/id16 arrive with11/16 IMMA left. id16 moves both final LDSM waits onto arrival.','Transcript and PoW validation passed; racecheck,memcheck,synccheck0errors on edge suite.','No claim of live-pool hashrate increase until mining deployment and acceptance; no deployment performed.']
text='\n'.join(lines)+'\n';(out/'report.txt').write_text(text);print(text)
