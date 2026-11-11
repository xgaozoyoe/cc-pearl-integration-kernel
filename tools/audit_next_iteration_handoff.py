"""Audit frozen next-iteration instructions; CPU only, no binary mutation."""
from pathlib import Path
import csv
import hashlib
import json
import re

root = Path(__file__).resolve().parents[1]
base = root / 'artifacts/fold-preamble'
out = root / 'artifacts/arrival-order'
v = json.loads((base/'manifest.json').read_text())['variants']['both_compact']
assert hashlib.sha256((base/'both_compact.cubin').read_bytes()).hexdigest() == v['sha256']
body = (base/'both_compact.sass').read_text().split('Function : '+v['entry'])[1].split('Function : ')[0]
ops = {int(pc, 16): op for pc, op in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;', body)}
assert ops[0x2a00] == 'BAR.SYNC.DEFER_BLOCKING R161, 0x180'
assert ops[0x1090] == 'LDSM.16.M88.4 R144, [R144]'
assert len(range(0xbb0, 0xdc0, 16)) == 33
assert len(range(0xdd0, 0xf40, 16)) == 23
copy_pcs = {0xf40, 0xfb0, 0x1020, 0x1060, 0x1070}
rows = []
for pc in list(range(0x2a00, 0x2a50, 16))+list(range(0xbb0, 0x10d0, 16)):
    op = ops[pc]
    if pc == 0x2a00:
        kind, hazard = 'named handoff', 'own late arrivals + peer early publications'
    elif 0x2a10 <= pc <= 0x2a40:
        kind, hazard = 'exit/ID/loop control', 'preserve old ID consumption, predicates and deferred-blocking controls'
    elif pc == 0xdc0:
        kind, hazard = 'future-copy skip branch', 'local iteration predicate; final compute iteration skips future copy'
    elif op == '@!PT LDS RZ, [RZ]':
        kind, hazard = 'never-executed shared-load placeholder', 'no shared access; preserve binary scheduling unless separately tested'
    elif pc in copy_pcs:
        assert 'LDGSTS.' in op
        kind, hazard = 'async global-to-shared copy', 'all readers of destination old-stage region must have finished LDSM'
    elif pc == 0x1080:
        kind, hazard = 'async group commit', 'local async group order; does not establish peer-ready visibility'
    elif 'LDSM.' in op:
        kind, hazard = 'operand shared-to-register read', 'producers of this operand region must have completed and published copies'
    else:
        assert not any(s in op for s in ('BAR.', 'STS ', 'LDS ', 'LDGSTS.', 'LDSM.'))
        kind, hazard = 'local address/predicate/constant work', 'no peer-data dependency; local register/scoreboard lifetimes still apply'
    rows.append(dict(pc=hex(pc), instruction=op, category=kind, dependency=hazard))
with (out/'next-iteration-instructions.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print('PASS:', len(rows), 'instructions classified; frozen cubin hash verified.')

# Each active warp copy instruction writes eight complete 64-byte rows.
# These are per-instruction overwrite dependencies, not whole-stage gates.
regions = []
for warp in range(12):
    for mat, j, pc in [('A', 0, 0xf40), ('A', 1, 0xfb0), ('A', 2, 0x1020),
                       ('B', 0, 0x1060), ('B', 1, 0x1070)]:
        lo = warp*8+j*96
        height = 256 if mat == 'A' else 192
        if lo >= height:
            continue
        assert lo//64 == (lo+7)//64
        readers = [w for w in range(12)
                   if (w % 4 if mat == 'A' else w//4) == lo//64]
        regions.append(dict(warp=warp, pc=hex(pc), matrix=mat,
                            rows=f'{lo}-{lo+7}', old_stage_readers=' '.join(map(str, readers))))
with (out/'copy-region-readers.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=regions[0].keys())
    w.writeheader()
    w.writerows(regions)
assert len(regions) == 56
print('PASS: 56 active per-warp copy operations classified by old-stage reader set.')
