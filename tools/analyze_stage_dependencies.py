"""CPU-only A/B stage hazard graph; no synchronization implementation changes."""
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / 'artifacts/arrival-order'
base = root / 'artifacts/fold-preamble'
meta = json.loads((base / 'manifest.json').read_text())['variants']['both_compact']
assert hashlib.sha256((base / 'both_compact.cubin').read_bytes()).hexdigest() == meta['sha256']
source = (root / 'artifacts/fold-defer-reduce/source.cu').read_text().split('} // namespace fold_defer_source8')[0]
for fragment in ('int wm=warp%4,wn=warp/4;',
                 'int logical=int(threadIdx.x)^(int(threadIdx.x>>3)&3);',
                 'threadIdx.x+j*THREADS<BM*64/16'):
    assert fragment in source, fragment

# Enumerate every 16-byte operand chunk. Swizzle is involutive and changes
# only the column chunk within a row; it preserves producer warp membership.
owners, readers = {}, {}
for mat, height in [('A', 256), ('B', 192)]:
    for tid in range(384):
        for j in range(3 if mat == 'A' else 2):
            physical = tid + j*384
            if physical >= height*4:
                continue
            logical = physical ^ ((physical >> 3) & 3)
            offset = logical*16
            assert offset ^ ((offset >> 3) & 0x30) == physical*16
            key = (mat, logical//4, logical % 4)
            assert key not in owners
            owners[key] = tid//32
    for row in range(height):
        consumers = {w for w in range(12)
                     if row//64 == (w % 4 if mat == 'A' else w//4)}
        for chunk in range(4):
            readers[mat, row, chunk] = consumers
assert owners.keys() == readers.keys()
assert len(owners)*16 == (256+192)*64

warps = []
for w in range(12):
    ready = {owners[k] for k, rr in readers.items() if w in rr}
    free = set().union(*(readers[k] for k, writer in owners.items() if writer == w))
    # Own readiness is already certified by this warp reaching its late point.
    deps = (ready | free) - {w}
    warps.append(dict(warp=w, next_operand_producers=sorted(ready),
                      old_stage_readers=sorted(free), external_dependencies=sorted(deps),
                      removable_external_dependencies=sorted(set(range(12)) - deps - {w})))
result = dict(baseline_sha256=meta['sha256'], chunks=len(owners), warps=warps,
              scope='Interior steady-state, unchanged whole-iteration gate, original copy ownership; excludes final/drain special cases.',
              model_warning='Source trace, fixed arrival timeline, no hardware wait or speedup measurement.',
              source_trace_models=[])
for kt in (16, 17, 46, 47):
    raw = np.concatenate([np.load(out/f'trace-k{kt}-r{r}.npy') for r in range(3)]).astype(np.int64)
    ts = (raw-raw[:, 0, 0, None, None]+2**31) % 2**32-2**31
    early, late = ts[:, :, 0], ts[:, :, 1]
    all_gate = np.maximum(early.max(axis=1)[:, None], late)
    narrowed = np.empty_like(all_gate)
    for w, row in enumerate(warps):
        narrowed[:, w] = np.maximum(early[:, row['external_dependencies']].max(axis=1), late[:, w])
    delta = all_gate-narrowed
    assert np.all(delta >= 0)
    result['source_trace_models'].append(dict(kt=kt,
        mean_gate_advance_cycles_per_warp=float(delta.mean()),
        mean_gate_advance_by_warp=delta.mean(axis=0).tolist(),
        warp_samples_advance_over64_pct=float(100*np.mean(delta > 64))))
(out/'stage-dependencies.json').write_text(json.dumps(result, indent=2)+'\n')
print('PASS: all', len(owners), '16-byte chunks have exactly one writer and their tile readers enumerated.')
for row in warps:
    print('warp', row['warp'], 'can omit external warps', row['removable_external_dependencies'])
for row in result['source_trace_models']:
    print(row)
