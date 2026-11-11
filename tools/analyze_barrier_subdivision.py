"""Static barrier subdivision model. No GPU launches or production changes."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
p=root/'artifacts/barrier-attribution';meta=json.loads((root/'artifacts/fold-preamble/manifest.json').read_text())['variants']['both_compact']
assert hashlib.sha256((root/'artifacts/fold-preamble/both_compact.cubin').read_bytes()).hexdigest()==meta['sha256']
owners={}
for operand,height in [('A',256),('B',192)]:
 for tid in range(384):
  logical=tid^((tid>>3)&3)
  for j in range(3 if operand=='A' else 2):
   if tid+j*384<height*4:
    row,col=logical//4+j*96,(logical%4)*16
    assert (row*64+col)^(((row*64+col)>>3)&0x30)==(tid+j*384)*16
    owners[operand,row,col]=tid//32
blocks=[]
for operand,count in [('A',4),('B',3)]:
 for block in range(count):
  producers=sorted({w for (op,row,col),w in owners.items() if op==operand and row//64==block})
  consumers=[w for w in range(12) if (w%4 if operand=='A' else w//4)==block]
  blocks.append(dict(operand=operand,block=block,producers=producers,consumers=consumers))
lines=['Static barrier subdivision audit; not a latency/performance prediction.', 'Frozen SASS SHA256 '+meta['sha256'], 'Current warp grid (A row group x B column group):']
for m in range(4):lines.append(' '.join(f'w{m+4*n:02}' for n in range(3)))
lines.append('Current copy ownership by logical64x64 operand block:')
for b in blocks:lines.append(f"{b['operand']}{b['block']}: producers={b['producers']}, consumers={b['consumers']}")
lines += ['', 'All-to-all early-publication, one late-wait barrier per consumer group:',
 'groups | main-loop IDs with two alternating phases | warp-level barrier instructions/K/CTA | ordered pairs gated by own-group late completion']
models=[]
for sizes in ([8,4],[6,6],[4,4,4],[3,3,3,3],[2]*6):
 g=len(sizes);pairs=sum(n*(n-1) for n in sizes)
 row=dict(sizes=sizes,ids=2*g,warp_barrier_instructions_per_iteration=12*g,late_peer_pairs=pairs)
 models.append(row);lines.append(f'{sizes} | {2*g} | {12*g} | {pairs}')
lines += ['', 'Each warp arrives early on EVERY other group barrier, then syncs late on its own. Each barrier still counts384 threads.',
 '3 groups:24 early warp-arrivals +12 late warp-syncs vs baseline12+12; +50% barrier instructions, not +50% total runtime.',
 'IDs can use1..6, leaving0 for prologue/epilogue; alternating phases avoid immediate ID reuse.',
 'Safety invariant: before consuming next stage or overwriting old stage, all own-group warps have reached late sync and all other-group warps have published completed copies+reads.',
 'This preserves existing data reuse/28672 unique logical copy bytes per CTA/K64. Requires implementation validation and phase-reuse proof before use.',
 'Own-group late-peer pairs fall68->36 for8+4->4+4+4; this is dependency-edge count, NOT a predicted speedup.',
 'A/B split is different:4 A cohorts with3 consumers each;3 B cohorts with4 consumers each. Current producers span other cohorts.',
 'Previous remapped cohort_ab/cohort_ba used two sequential blocking barriers per warp. Intersection propagates late dependencies across all12 warps; do not repeat as a new design.',
 'Readiness and overwrite permission must both be protected.14 logical ready/free objects (7x2) would become28 if naively doubled by stage, exceeding16 named IDs.',
 'Do not infer fewer IDs suffice without a safe lifetime/reuse protocol. More named barriers alone does not reduce wait.',
 'Recommended next experiment: existing8+4 vs4+4+4 early-arrive/late-sync, same operands/layout/MMA, include same-structure compiled baseline and frozen-native comparison.',
 'Use2 extra early publication groups only if actual saved tail wait outweighs added arrive/ID instructions; no performance claim yet.']
(p/'subdivision.json').write_text(json.dumps(dict(blocks=blocks,models=models),indent=2)+'\n')
(p/'subdivision.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
