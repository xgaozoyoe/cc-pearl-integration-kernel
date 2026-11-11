"""Audit static LDSM -> IMMA distances in the measured A-stream baseline.

Distances are instruction counts, not cycles. Only the straight-line IMMA
body is analyzed; no dynamic warp phase or SASS control-word inference.
"""
from pathlib import Path
import collections
import json
import re
import argparse

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--sass', type=Path, default=root/'artifacts/rank128-astream-ablation.sass')
parser.add_argument('--namespace', default='astream_ablation_full')
parser.add_argument('--output', type=Path, default=root/'artifacts/operand-lifetimes.json')
args = parser.parse_args()
sass = args.sass.resolve()
section = next(s for s in sass.read_text().split('Function : ')[1:]
               if args.namespace+'6kernel' in s.splitlines()[0] or args.namespace+'::kernel' in s.splitlines()[0])
rows = [(int(pc, 16), ins) for pc, ins in
        re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;', section)]
mma = {}
for i, (pc, ins) in enumerate(rows):
    if ins.startswith('IMMA.16832.S8.S8 '):
        regs = [int(x) for x in re.findall(r'\bR(\d+)', ins)]
        assert len(regs) == 4
        mma[i] = dict(A=set(range(regs[1], regs[1]+4)),
                      B=set(range(regs[2], regs[2]+2)), C=regs[3], D=regs[0])
assert len(mma) == 64
loads = []
for i, (pc, ins) in enumerate(rows):
    match = re.match(r'LDSM\.16\.M88\.([24]) R(\d+),', ins)
    if not match or i > max(mma):
        continue
    regs = set(range(int(match[2]), int(match[2])+int(match[1])))
    # Stop at next LDSM overwriting this fragment. Current loop has no MOV
    # copies of loaded operand fragments; consumers read these registers.
    end = next((j for j in range(i+1, max(mma)+1)
                if (m := re.match(r'LDSM\.16\.M88\.([24]) R(\d+),', rows[j][1]))
                and regs & set(range(int(m[2]), int(m[2])+int(m[1])))), max(mma)+1)
    overwrite = next((j for j in mma if j > i and regs & set(range(mma[j]['D'], mma[j]['D']+4))), max(mma)+1)
    users = [(j, role) for j in mma if i < j < end and j <= overwrite
             for role in ('A', 'B') if regs & mma[j][role]]
    if not users:
        continue
    first, role = users[0]
    previous = [j for j in mma if j < i and (regs & mma[j]['A'] or regs & mma[j]['B'])]
    loads.append(dict(load_pc=hex(pc), register_start=int(match[2]), register_count=int(match[1]), role=role,
                      first_use_pc=hex(rows[first][0]),
                      instructions_between=first-i-1,
                      independent_IMMA_between=sum(i < j < first for j in mma),
                      previous_operand_use_pc=hex(rows[previous[-1]][0]) if previous else None,
                      IMMA_after_previous_use_before_load=sum(previous[-1] < j < i for j in mma) if previous else None,
                      consumer_count=len(users)))
last_c = {}
recurrences = []
for j, op in mma.items():
    c = op['C']
    if c in last_c:
        prev = last_c[c]
        recurrences.append(dict(previous_pc=hex(rows[prev][0]), pc=hex(rows[j][0]),
                                independent_IMMA_between=sum(prev < k < j for k in mma)))
    last_c[op['D']] = j
result = dict(source=str(sass.relative_to(root)), loads=loads,
              C_recurrences=recurrences,
              caveats=['Static instructions, not cycle distances.',
                       'No dynamic evidence of warp phase alignment.',
                       'C recurrence counts cover the unrolled body only, not the loop backedge.'])
assert len(recurrences) == 32
assert sum(x['consumer_count'] for x in loads) == 128
profile = next(x for x in json.loads((root/'artifacts/rank128-astream-ablation-profile/pc-stalls.json').read_text())
               if Path(x['file']).name == 'v0.sass.csv')
samples = collections.Counter()
for row in profile['pcs']:
    samples[row['pc_offset']] += row['samples'].get('stall_short_sb', 0)
for load in loads if args.namespace == 'astream_ablation_full' else []:
    load['short_scoreboard_samples_at_first_use'] = samples[load['first_use_pc']]
    load['percent_of_all_short_scoreboard_samples'] = 100*samples[load['first_use_pc']]/profile['sample_columns']['stall_short_sb']
args.output.write_text(json.dumps(result, indent=2)+'\n')
for item in loads:
    print(item)
print('C recurrence distance histogram:', dict(collections.Counter(x['independent_IMMA_between'] for x in recurrences)))
