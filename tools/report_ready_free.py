"""Summarize ready/free protocol experiment without claiming isolated SASS causality."""
from pathlib import Path
import json,re
p=Path(__file__).resolve().parents[1]/'artifacts/ready-free'
m=json.loads((p/'manifest.json').read_text());lines=['GPU0 only;8192x8192 region,K4096,rank128. CUDA event kernel times;not full-grid or pool rate.',
'Two independent runs;200samples/variant/run,16launches/sample,10 rotating blocks;unlocked clocks.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries']
 lines += [f'Run{run}: name mean_ms median_ms p95_ms delta_native_pct delta_source_pct']
 for n,v in s.items():
  lines.append(f"{n:16} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.4f} {(v['mean_ms']/s['source_control']['mean_ms']-1)*100:+.4f}")
 lines.append(f"deferred vs late time delta:{(s['split_deferred']['mean_ms']/s['split_late']['mean_ms']-1)*100:+.4f}%")
for n,v in m['variants'].items():
 if n=='native':s=(p.parent/'fold-preamble/both_compact.sass').read_text()
 else:s=(p/(n+'.sass')).read_text()
 s=s.split('Function : '+v['entry'])[1].split('Function : ')[0]
 rows={int(a,16):b for a,b in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)}
 counts={op:sum(op in x for x in rows.values()) for op in ['IMMA.','LDSM.','LDGSTS.E.','BAR.ARV','BAR.SYNC','ARRIVES.LDGSTSBAR','ATOMS.ARRIVE','MEMBAR.ALL.CTA']}
 lines.append(n+' static SASS '+json.dumps(counts)+' smem='+str(v['smem']))
 lastmma=max(pc for pc,x in rows.items() if 'IMMA.' in x)
 pub=[pc for pc,x in rows.items() if 'BAR.ARV' in x or 'ATOMS.ARRIVE' in x]
 lines.append(' last IMMA='+hex(lastmma)+' publication='+','.join(hex(x) for x in pub))
assert len(json.loads((p/'validation.json').read_text()))==48
assert len(json.loads((p/'pow-validation.json').read_text()))==24
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck.log').read_text()
lines += ['48 transcript +24 transcript/PoW comparisons PASS;racecheck0 hazards/errors/warnings.',
'All168regs,0spills/local,384threads,1CTA/SM. Dynamicshared:69632B native/control,69664B split.',
'Both split variants:4mbarriers (ready/free per physical slot),eachcount384,phase(k/2)&1.',
'One async-ready arrival/thread/copied stage,one free arrival/thread/read stage. No extra normal arrival on ready.',
'late waits on free(current) and ready(next) at loop end. deferred waits on free before overwrite and ready before LDSM.',
'No main-loop CTA barrier added. Shared mbarriers are SM80+ facilities, not Hopper transaction barriers.',
'Important confound:source_control is recompiled and publishes after64IMMA;frozen native after48IMMA. Split SASS also schedules differently. Timing does not isolate barrier instruction latency.',
'No hardware counters collected;no claim of measured tensor utilization or polling stall proportions.',
'No production libraries changed,no live mining. No full-grid benefit claimed.',
'PTX semantics reference:https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
