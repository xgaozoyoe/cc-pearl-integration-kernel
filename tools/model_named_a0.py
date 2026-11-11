"""CPU protocol model: two alternating named-barrier IDs for early A[32:64] reuse by warps4..7.

Not a CUDA correctness test: abstracts instruction completion and memory ordering.
Checks participant counts, phase reuse, read/write ordering, and deadlock under
random and fixed-priority warp schedules. Does not mutate any mining binary.
"""
from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/named-a0-design'
OUT.mkdir(exist_ok=True)
R = {('A', i): {i, i+4, i+8} for i in range(4)}
R.update({('B', i): set(range(i*4, i*4+4)) for i in range(3)})
chunks = {w: [(mat, j, (w*8+j*96)//64)
              for mat, height, rounds in [('A', 256, 3), ('B', 192, 2)]
              for j in range(rounds) if w*8+j*96 < height] for w in range(12)}

def run(seed, policy):
    rng = random.Random(seed)
    programs = []
    for w in range(12):
        p = []
        for k in range(64):
            if k:
                if k < 63 and 4 <= w < 8:
                    p += [('sync', 1+((k-1)&1), k-1, 6), ('copy', k+1, [('A', 0, 0)])]
                p += [('sync', (4+(w >= 8)) ^ (2*((k-1) % 2)), k-1, 12)]
            if k < 63:
                parts = [c for c in chunks[w] if not (k and 4 <= w < 8 and c[:2] == ('A', 0))]
                p += [('copy', k+1, parts)]
            p += [('read', k), ('complete', k+1),
                  ('arrive', ((4+(w >= 8)) ^ (2*(k % 2))) ^ 1, k, 12)]
            if w in (0, 8) and k < 62:
                p += [('arrive', 1+(k&1), k, 6)]
            p += [('tail', k)]
        p += [('sync', (4+(w >= 8)) ^ 2, 63, 12)]
        programs.append(p)
    pc = [0]*12
    blocked = [None]*12
    barriers = {}
    completed = set()
    last_read = [-1]*12
    copied = {(w, mat, j, 0) for w in range(12) for mat, j, _ in chunks[w]}
    ready_k = [0]*12
    arrivals = 0
    while any(pc[w] < len(programs[w]) for w in range(12)):
        active = [w for w in range(12) if pc[w] < len(programs[w]) and
                  (blocked[w] is None or blocked[w] in completed)]
        assert active, ('deadlock', seed, policy, pc, blocked)
        w = rng.choice(active) if policy == 'random' else (min(active) if policy == 'ascending' else max(active))
        blocked[w] = None
        event = programs[w][pc[w]]
        pc[w] += 1
        op = event[0]
        if op in ('arrive', 'sync'):
            _, bid, generation, count = event
            initial = 0 if bid in (1, 4, 5) else 1
            state = barriers.setdefault(bid, [initial, set()])
            assert state[0] == generation, ('phase mismatch', w, event, state)
            assert w not in state[1], ('double arrival', w, event)
            state[1].add(w)
            arrivals += 1
            token = (bid, generation)
            if op == 'sync':
                blocked[w] = token
            if len(state[1]) == count:
                completed.add(token)
                state[0] += 2
                state[1] = set()
        elif op == 'copy':
            _, copied_k, parts = event
            for mat, j, region in parts:
                assert all(last_read[r] >= copied_k-2 for r in R[mat, region]), ('overwrite hazard', w, event, last_read)
                key = (w, mat, j, copied_k)
                assert key not in copied, ('duplicate copy', key)
                copied.add(key)
        elif op == 'complete':
            copied_k = event[1]
            if copied_k < 64:
                assert all((w, mat, j, copied_k) in copied for mat, j, _ in chunks[w])
                ready_k[w] = copied_k
        elif op == 'read':
            k = event[1]
            # Conservatively check full stage publication rather than individual loads.
            assert all(v >= k for v in ready_k), ('read before ready', w, k, ready_k)
            last_read[w] = k
    assert all(not state[1] for state in barriers.values()), barriers
    assert len(copied) == 56*64
    assert arrivals == 12*2*64+6*62
    return arrivals

for policy in ('ascending', 'descending'):
    run(0, policy)
for seed in range(1000):
    run(seed, 'random')
result = dict(schedules=1002, random_seeds=1000, rounds=64,
              old_arrivals_per_cta=12*2*64, new_arrivals_per_cta=6*62,
              checks=['no duplicate participants', 'phase progression', 'no partial barrier at exit',
                      'no modeled overwrite/read-before-ready', 'same copy count', 'no modeled deadlock'],
              limitations='Abstract event model, not CUDA/SASS memory-order validation, exhaustive proof, or performance evidence.')
(OUT/'model.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
