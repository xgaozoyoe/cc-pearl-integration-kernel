"""Copy-count x barrier ablation. Never interpret traffic removal as pure skew removal."""
from pathlib import Path
import json,argparse
p=argparse.ArgumentParser();p.add_argument('capture',nargs='?',type=Path);args=p.parse_args();out=Path(__file__).resolve().parents[1]/'artifacts/copy-skew';names=['native','equal_copy','no_barrier','both']
lines=['Offline wrong-arithmetic diagnostic, no code motion/recompilation. Frozen fastest cubin.', 'Factor1 removes third A LDGSTS at1020. Warps0..7 lose one16B/lane transfer; all warps now execute4copy instructions per copied stage.', 'A stage loses4096B/28672B=14.286% logical transfer traffic. The actual L2 change must be measured.', 'Factor2 removes16d0 BAR.ARV and2a00 BAR.SYNC; other synchronization retained.', 'All IMMA/LDSM, loop, registers/shared allocation, waits and fixed stall slots remain.', 'Skipped shared regions can be stale/uninitialized, but values only enter arithmetic, never address generation. Output is invalid.', 'This tests a candidate contribution and its interaction with barriers; it does NOT establish which warp arrived last.', 'Less traffic and different numerical switching are confounds. A positive interaction is a clue, not proof of copy imbalance as cause.', '']
def compare(t,label):
 lines.append(label+': '+json.dumps(t))
 lines.append(f'Copy removal gain with barriers {100*(1-t["equal_copy"]/t["native"]):.4f}%; without barriers {100*(1-t["both"]/t["no_barrier"]):.4f}%.')
 lines.append(f'Barrier removal gain original copy {100*(1-t["no_barrier"]/t["native"]):.4f}%; equal copy {100*(1-t["both"]/t["equal_copy"]):.4f}%.')
 interaction=t['native']-t['equal_copy']-t['no_barrier']+t['both']
 lines.append(f'Interaction (T00-T10)-(T01-T11):{interaction:.9f} {label}; {100*interaction/t["native"]:.4f}% of native. Positive means part of copy-removal benefit overlaps barrier-removal benefit.')
if (out/'events.json').exists():
 e=json.loads((out/'events.json').read_text());lines.append('Unlocked events:200samples/variant,16launches/sample,10rotating blocks. Region8192x8192x4096, not full mining grid.')
 for n in names:lines.append(n+' '+json.dumps(e['summaries'][n]))
 compare({n:e['summaries'][n]['mean_ms'] for n in names},'unlocked ms')
if args.capture:
 issues=json.loads((args.capture/'imma-issue.json').read_text());s=json.loads((args.capture/'summary.json').read_text())
 for i,n in enumerate(names):
  assert all(x['count_matches_sass'] for x in issues[str(i)]['launches'])
  lines.append(n+' '+json.dumps(issues[str(i)]['means']));lines.append(json.dumps({k:v['mean'] for k,v in s[str(i)]['metrics'].items()}))
 compare({n:issues[str(i)]['means']['kernel_ms'] for i,n in enumerate(names)},'base-clock ms')
 compare({n:1/issues[str(i)]['means']['imma_per_SM_elapsed_cycle'] for i,n in enumerate(names)},'SM elapsed cycles per IMMA')
 (args.capture/'copy-skew-report.txt').write_text('\n'.join(lines)+'\n')
else:(out/'report.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
