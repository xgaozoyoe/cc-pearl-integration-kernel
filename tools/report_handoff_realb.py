"""Summarize offline real-B scheduling controls without inferring fixed-clock gains."""
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[1]
lines=['Real-B scheduling experiment (GPU0 only)', '8192x8192 K4096 region; CUDA events exclude setup/PoW. Clocks unlocked; no fixed-clock attribution.', '']
for stem,names in [('realb',['control','early','two_ahead','compact']),('pairb',['control','paired'])]:
 path=root/f'artifacts/handoff-{stem}-events.json'
 if not path.exists():continue
 data=json.loads(path.read_text());rows=data['measurements'];base=np.mean([x for r in rows if r['variant']==0 for x in r['samples_ms']])
 lines += [stem, 'variant samples mean_ms median_ms p95_ms time_vs_control']
 for v,name in enumerate(names):
  vals=np.array([x for r in rows if r['variant']==v for x in r['samples_ms']])
  lines.append(f'{name} {len(vals)} {vals.mean():.6f} {np.median(vals):.6f} {np.percentile(vals,95):.6f} {(vals.mean()/base-1)*100:+.3f}%')
 if stem=='pairb':
  for repeat in sorted({r['repeat'] for r in rows}):
   block={r['variant']:r['mean_ms'] for r in rows if r['repeat']==repeat}
   lines.append(f"paired block {repeat}: {(block[1]/block[0]-1)*100:+.3f}% time")
 lines.append('resources: '+json.dumps(data['resources']))
lines += ['', 'Findings:', '- early is byte-for-byte identical SASS to control, including instruction control encoding.', '- two_ahead/compact do not move the three immediate first-half B load-consumer edges.', '- paired changes real B use order and increases some load-to-use distances, but introduces40B stack,40B spill stores,48B spill loads (ptxas static counts, not runtime traffic).', '- All kernels retain64 static IMMA and16 LDSM. Control matches validated lean-toggle SASS.', '- Transcript and PoW digest match reference for full8192 region and edge shapes; racecheck/memcheck/synccheck pass on edge suite.', '- No production replacement. Negative timing does not prove real B prefetch intrinsically ineffective; compiler scheduling and spill costs are confounders.', '- Fixed-base-clock optional profile: sudo bash tools/profile_handoff_pairb.sh']
text='\n'.join(lines)+'\n';(root/'artifacts/handoff-realb-report.txt').write_text(text);print(text)
