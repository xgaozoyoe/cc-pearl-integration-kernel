"""Audit frozen native barrier participants and arrival conditions. CPU only."""
from pathlib import Path
import json,hashlib,re
root=Path(__file__).resolve().parents[1];p=root/'artifacts/fold-preamble'
v=json.loads((p/'manifest.json').read_text())['variants']['both_compact']
assert hashlib.sha256((p/'both_compact.cubin').read_bytes()).hexdigest()==v['sha256']
s=(p/'both_compact.sass').read_text().split('Function : '+v['entry'])[1].split('Function : ')[0]
ops={int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)}
assert ops[0x1620]=='LOP3.LUT R164, R161, 0x1, RZ, 0x3c, !PT'
assert ops[0x16d0]=='BAR.ARV R164, 0x180'
assert ops[0x2a00]=='BAR.SYNC.DEFER_BLOCKING R161, 0x180'
assert ops[0x2a20]=='LOP3.LUT R161, R161, 0x2, RZ, 0x3c, !PT'
assert sum('IMMA.' in x for pc,x in ops.items() if 0x16d0<pc<0x17e0)==16
assert sum('IMMA.' in x for pc,x in ops.items() if 0xbb0<=pc<0x16d0)==48
rounds=[]
for k in range(64):
 groups=[list(range(8)),list(range(8,12))]
 for g in range(2):
  own=(4+g)^(2*(k%2));peer=own^1
  early=groups[1-g];late=groups[g]
  assert (len(early)+len(late))*32==384
  rounds.append(dict(k_iteration=k,barrier_id=own,waiting_warps=late,early_arrival_warps=early,
                     early_arrival_pc='0x16d0',late_arrival_pc='0x2a00',
                     fold=(k%2==1),fold_mode=('none' if k%2==0 else 'store' if k<32 else 'rotate_xor')))
out=root/'artifacts/barrier-attribution';out.mkdir(exist_ok=True)
(out/'native-participants.json').write_text(json.dumps(rounds,indent=2)+'\n')
issue=json.loads((root/'artifacts/wait-vs-handoff/profile/imma-issue.json').read_text())
base=issue['0']['means']['imma_per_SM_elapsed_cycle']
lines=[
'Frozen-native barrier semantics audit. No new GPU experiment or per-warp timestamps.',
'SHA256 '+v['sha256'],
'Group0=warps0..7,256threads; Group1=warps8..11,128threads.',
'Each named barrier counts384threads, combining peer early arrivals with own late sync arrivals.',
'Even K index: ID4=G1early+G0late; ID5=G0early+G1late. No transcript fold.',
'Odd K index: ID6=G1early+G0late; ID7=G0early+G1late. Transcript fold.',
'64iterations:32without fold,16initial transcript stores,16rotate-XOR updates.',
'',
'Per warp:',
'copy future stage ->48IMMA/interleavedLDSM ->16c0 cp.wait0 ->16d0 peer BAR.ARV',
'->16tailIMMA ->optional transcript ->2a00 own BAR.SYNC ->2a30 deferred blocking ->next iteration.',
'16d0 additionally waits SB1/SB2 for last LDSM; arrive itself is not a wait for peer arrivals.',
'Publish certifies this warp future copies completed and old-stage reads completed.',
'Tail IMMA uses registers; remaining transcript accesses use a separate shared region.',
'Other warps do not need this warp tail arithmetic to consume operand stages once its publication is complete.',
'But own-group BAR.SYNC arrival comes after those instructions, so current protocol includes that extra rendezvous.',
'',
'Abstract arrival-count model, ignoring memory-order completion and barrier hardware latency:',
'P[w]=early-arrival time; L[w]=late-sync-arrival time; P[w]<=L[w].',
'G0 release threshold=max(max(P[8:12]),max(L[0:8])).',
'G1 release threshold=max(max(P[0:8]),max(L[8:12])).',
'For own-group warp w, same-group extra arrival wait over all-publication readiness:',
'max(0,max(L[own])-max(L[w],max(P[all]))).',
'This is a counterfactual bound under the same arrival timeline, NOT an achieved speedup.',
'Actual resume also includes synchronization memory ordering, hardware completion and scheduling.',
'Transcript stores preceding BAR.SYNC may participate in ordering; existing data cannot separate this from arrival skew.',
'',
'Existing fixed-clock-policy evidence, IMMA/SMelapsedcycle change:',
]
for i,n in [('1','remove explicit cp.wait only'),('2','remove named arrive+sync'),('3','remove both')]:
 lines.append(f'{n}: {100*(issue[i]["means"]["imma_per_SM_elapsed_cycle"]/base-1):+.6f}%')
lines += [
'Named removal is intentionally racy: diagnostic headroom, not a valid optimization.',
'No-copy-wait result does NOT prove copies never delay any warp; it measures marginal effect in this configuration.',
'',
'What is still unknown:',
'Which warp last publishes, which warp last late-arrives, and how often each determines release.',
'How that differs for no-fold/store/rotate-XOR rounds, CTA boundary types and scheduler placement.',
'How much of observed delay is count arrival, ordering, barrier machinery or rescheduling.',
'',
'Next measurement requirements (not yet implemented):',
'Sparse frozen-binary per-warp P/L timestamps, tagged by CTA and K index; classify peer-publication vs own-late bottleneck.',
'Use same-SM clock, keep original BARs and scheduling guards; ensure clock timestamps obey the intended dependencies.',
'Exit timestamp must be after actual deferred wait completion, NOT merely after BAR.SYNC issue.',
'Measure instrumentation overhead and compare multiple sampling rates; no full-grid trace stores inside every iteration.',
'Previous recompiled clock64 tracer rejected: it changed48/16MMA around arrival to56/8 or64/0.',
'Do not use those traces as evidence for this frozen baseline.',
'',
'PTX semantic reference: https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier',
]
(out/'native-waits.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
