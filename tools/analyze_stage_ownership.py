"""Static K64 ownership audit; does not launch or modify a CUDA kernel.

Models Layout6 copy assignment and warp64x64 reads in fold-defer source8.
SASS hash anchors the measured baseline; this is not a timing simulation.
"""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/barrier-attribution'
meta = json.loads((ROOT / 'artifacts/fold-preamble/manifest.json').read_text())['variants']['both_compact']
assert hashlib.sha256((ROOT / 'artifacts/fold-preamble/both_compact.cubin').read_bytes()).hexdigest() == meta['sha256']

def swizzle(p):
    return p ^ ((p >> 3) & 0x30)

owners = {}
for operand, height in [('A', 256), ('B', 192)]:
    for thread in range(384):
        logical = thread ^ ((thread >> 3) & 3)
        for j in range(3 if operand == 'A' else 2):
            index = thread + j * 384
            if index >= height * 4:
                continue
            row, col = logical // 4 + j * 96, (logical % 4) * 16
            assert swizzle(row * 64 + col) == index * 16
            key = (operand, row, col)
            assert key not in owners
            owners[key] = thread // 32
assert len(owners) == (256 + 192) * 4

reads = []
for warp in range(12):
    wm, wn = warp % 4, warp // 4
    reads.append({(operand, row, col)
                  for operand, first in [('A', wm * 64), ('B', wn * 64)]
                  for row in range(first, first + 64)
                  for col in range(0, 64, 16)})

rows = []
for warp in range(12):
    produced = {key for key, owner in owners.items() if owner == warp}
    rows.append(dict(warp=warp, wm=warp % 4, wn=warp // 4,
                     ready_producers=sorted({owners[key] for key in reads[warp]}),
                     overwrite_readers=[w for w in range(12) if produced & reads[w]],
                     copy_bytes=len(produced) * 16))

def partition_cost(groups):
    # Private storage per group removes cross-group sharing by duplication.
    return sum(len(set().union(*(reads[w] for w in group))) * 16 for group in groups)

partitions = []
for mask in range(1, 1 << 11):
    # Warp11 belongs to right group: enumerate every nonempty bipartition once.
    left = [w for w in range(11) if mask & (1 << w)]
    right = [w for w in range(12) if w not in left]
    partitions.append((partition_cost([left, right]), left, right))
best = min(partitions)
balanced = min(p for p in partitions if len(p[1]) == 6)
lines = [
    'Static dependency audit, not a GPU performance measurement.',
    'Baseline SHA256: ' + meta['sha256'],
    'CTA256x192, warp64x64, K64, 12warps; padded physical stages unchanged.',
    'P=producer warps needed before reads; R=reader warps needed before overwrite.',
]
for row in rows:
    lines.append(f"warp{row['warp']:2}: P={row['ready_producers']} R={row['overwrite_readers']}")
lines += [
    '',
    'Logical reuse: each A64x64 block has3 reader warps (same wm); each B64x64 block has4 (same wn).',
    'Current linear copy assignment distributes those blocks across other warps.',
    'Thus local cp.async.wait cannot replace cross-warp publication or reader completion.',
    'Current peer early-arrive + own late-sync also waits for own-group late arrivals.',
    'That late rendezvous is stronger than all-warp copy/read completion; removing it safely still requires a completion protocol.',
    'Earlier mbarrier polling was tested and slower; this audit does not establish a cheaper protocol.',
    '',
    f'Current unique copy bytes/K64/CTA: {len(owners)*16}',
    f'Private warp operands (no cross-warp reuse): {partition_cost([[w] for w in range(12)])} bytes',
    f'Private current8+4 groups: {partition_cost([list(range(8)), list(range(8,12))])} bytes',
    f'Minimum private2-group bytes over2047 partitions: {best}',
    f'Minimum private6+6-group bytes: {balanced}',
    'These are logical cp.async bytes, NOT measured L2 bytes; caches and transaction effects are not modeled.',
    '',
    'Candidate requiring further design: reassign copy ownership to A-row/B-column reuse cohorts.',
    'This can preserve28672 copy bytes and remove unrelated producer dependencies, but requires both ready and reuse handoffs.',
    '3-warp A cohorts and4-warp B cohorts overlap: local barriers can indirectly propagate waits; no gain is assumed.',
    'Simple double-buffer K128 batching requires114688 operand bytes alone, exceeding SM89 per-block99KiB.',
    'Therefore halving barrier frequency that way does not fit the current CTA, even without transcript storage.',
]
grids = ROOT / 'artifacts/cta-handoff/grids.json'
if grids.exists():
    measurements = json.loads(grids.read_text())['measurements']
    lines += ['', 'Previous full-CTA replacement: offline full-grid wall means, unlocked clocks.']
    for name in ['native', 'cta_sync']:
        values = [x for row in measurements if row['name'] == name for x in row['samples_ms']]
        mean = statistics.mean(values)
        lines.append(f'{name}: n={len(values)},mean={mean:.6f}ms,median={statistics.median(values):.6f}ms,equivalent={2**46/(mean/1000)/1e12:.6f}TH/s')
OUT.mkdir(exist_ok=True)
(OUT / 'ownership.json').write_text(json.dumps(dict(baseline_sha256=meta['sha256'], warps=rows, minimum_private_two_groups=best), indent=2) + '\n')
(OUT / 'ownership.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
