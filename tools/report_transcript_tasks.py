"""Report fine-grained independently executed transcript tasks."""
from pathlib import Path
import json,re,ast
p=Path(__file__).resolve().parents[1]/'artifacts/transcript-tasks';lines=['GPU0 only,8192x8192 region,K4096,rank128. Kernel CUDA-event timings,not fullgrid/pool throughput.',
'2 independent runs,200samples/variant/run,16launches/sample,10rotating blocks;unlocked clocks.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries']
 lines.append(f'Run{run}: name mean_ms median_ms p95_ms delta_native_pct')
 for n,v in s.items():lines.append(f"{n:16} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.3f}")
 lines.append(f"fine vs late time delta:{(s['consumer_fine']['mean_ms']/s['consumer_late']['mean_ms']-1)*100:+.3f}%")
 resources=json.loads((p/f'events-run{run}.json').read_text())['resources'];lines.append('resources [regs,local_bytes,static_shared,CTAs/SM] '+json.dumps(resources))
for name in ('consumer_late','consumer_fine'):
 r={int(a,16):b for a,b in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',(p/(name+'.sass')).read_text())}
 mma=[pc for pc,x in r.items() if x.startswith('IMMA.')];assert len(mma)==64
 pubs=[(pc,sum(m<pc for m in mma)) for pc,x in r.items() if 'ATOMS.ARRIVE' in x and '+0x12800]' in x]
 assert [n for pc,n in pubs]==([64]*4 if name=='consumer_late' else [40,48,56,64])
 lines.append(name+' task publications after MMA counts:'+str(pubs))
 lines.append((p/(name+'.build.log')).read_text().strip())
assert len(json.loads((p/'pow-validation.json').read_text()))==24
coverage=set()
for filename in ('racecheck.log','racecheck-tail.log'):
 text=(p/filename).read_text()
 assert 'ERROR:' not in text and 'WARNING:' not in text
 for line in text.splitlines():
  if line.startswith("{'name':"):
   r=ast.literal_eval(line);coverage.add((r['name'],r['m'],r['n'],r['seed']))
assert len(coverage)==48,len(coverage)
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck-tail.log').read_text()
lines+=['Normal transcript validation48comparisons PASS;PoW24comparisons PASS.',
'Racecheck original run completed46cases without reported hazards but timed out at240s;remaining2cases separately completed with0hazards/errors/warnings. No single full-run completion summary.',
'12MMA producers+1independent consumer;4tile tasks depend only on their producer warp and corresponding final MMA group.',
'Fine variant changes actual SASS publication points,not just source organization.',
'Consumer has fixed group-major/producer order,NOT ready-any scheduling;head-of-line waiting remains a limitation.',
'Explicit task shared traffic1.5MiB/CTA/K4096,excluding flags and existing transcript stores. Not measured DRAM traffic.',
'13warp variants incur register spills;therefore not acceptable production candidates. Timing includes spills,shared transport,notification and compiler schedule changes.',
'These timings cannot establish a general limit on fine-grained transcript dependency scheduling.',
'No production libraries changed,no live mining,no hardware-counter or full-grid speedup claims.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
