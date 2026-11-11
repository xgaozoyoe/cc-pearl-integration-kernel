"""Enumerate tile-level dependencies without changing the kernel or ownership."""
from pathlib import Path
import csv
import runpy

ROOT = Path(__file__).resolve().parents[1]
# Recheck frozen binary hash, actual barrier PCs and source ownership assertions.
runpy.run_path(str(ROOT / 'tools/audit_transcript_warp_tasks.py'))
out = ROOT / 'artifacts/transcript-warp-tasks'
rows = []
for warp in range(12):
    for row in range(4):
        for group in range(4):
            elements = set()
            for lane in range(32):
                for half in range(2):
                    for v in range(4):
                        # c[row][group][half][v], native m16n8 accumulator layout.
                        i = lane // 4 + 8 * (v // 2)
                        j = half * 8 + 2 * (lane % 4) + v % 2
                        assert (i, j) not in elements
                        elements.add((i, j))
            assert elements == {(i, j) for i in range(16) for j in range(16)}
            rows.append(dict(
                warp=warp, row=row, group=group,
                cta_tile_row=4*(warp % 4)+row,
                cta_tile_col=4*(warp // 4)+group,
                input=f'c[{row}][{group}][0:2][0:4] across own 32 lanes',
                producer_warp=warp, consumer_warp=warp,
                cross_warp_accumulator_dependency='none',
                blocking_named_barrier_before_current_fold='none in steady-state iteration',
                blocking_named_barrier_after_current_fold='0x2a00',
            ))
assert len(rows) == 192
assert len({(r['cta_tile_row'], r['cta_tile_col']) for r in rows}) == 192
with (out / 'subtask-dependencies.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
(out / 'subtask-boundaries.txt').write_text('''Scope: frozen fastest baseline; 12 warps, 384 threads. No kernel modification.
192 independent tile reductions per fold, each uses 256 accumulator values,
8 registers per lane across its own 32 lanes. All 256 elements enumerated once.
Source: artifacts/fold-defer-reduce/source.cu, first namespace only.
128-132: ownership; 188: MMA; 228-234: local XOR and own-warp REDUX;
236-238: private transcript state; 244: blocking named stage handoff;
247-250: final CTA barrier and collective output reads.

Frozen SASS (not source order) determines current instruction order:
16c0 copy wait -> 16d0 nonblocking peer arrive -> tail MMA -> transcript
-> 2a00 blocking own-group stage handoff -> next K iteration.
The copy wait and arrive's LDSM scoreboard wait can delay reaching transcript.
They do not wait for peer accumulator completion. No per-transcript named
barrier exists to narrow from all producers to a subset of producers.
The previous iteration's handoff can indirectly delay future MMA/transcript.

The 16 tiles are data-independent tasks, but remain instructions in the same
warp. A tile becoming ready does not create an independently schedulable warp.
Changing barrier participant sets alone cannot change that execution property.
Early tile reduction would require instruction scheduling changes; separate
warp execution would require changing ownership/data transport. Neither is
silently substituted for the requested barrier-only experiment here.

Do not shrink 2a00 based on transcript ownership: it protects shared A/B stage
readiness and reuse, whose copy/read producer sets are distinct from tile owners.
The final CTA barrier also protects cross-warp output reads, not local XOR.
This is static dependency evidence, not a new performance measurement.
''')
print('PASS: 192 tile subtasks; 256 unique accumulator elements per tile.')
