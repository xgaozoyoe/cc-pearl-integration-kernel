"""Audit the user's producer-warp -> transcript-subtask partition on fastest SASS.
No instruction relocation, no GPU workload, no production modifications.
"""
from pathlib import Path
import json,hashlib,re
root=Path(__file__).resolve().parents[1];out=root/'artifacts/transcript-warp-tasks';out.mkdir(exist_ok=True)
p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];blob=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(blob).hexdigest()==v['sha256']
s=(p/'both_compact.sass').read_text().split('Function : '+v['entry'])[1].split('Function : ')[0]
r={int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)}
assert r[0x16c0]=='DEPBAR.LE SB0, 0x0'
assert r[0x16d0]=='BAR.ARV R164, 0x180'
assert r[0x2a00]=='BAR.SYNC.DEFER_BLOCKING R161, 0x180'
assert r[0x2a50]=='BAR.SYNC.DEFER_BLOCKING 0x0'
assert not any('BAR.SYNC' in x for pc,x in r.items() if 0xbb0<=pc<0x2a00)
assert sum('IMMA.' in x for pc,x in r.items() if 0xbb0<=pc<0x2a00)==64
assert sum('REDUX.XOR' in x for pc,x in r.items() if 0x17e0<=pc<0x2a00)==32 #16 per mutually exclusive store/update branch
source=(root/'artifacts/fold-defer-reduce/source.cu').read_text().split('} // namespace fold_defer_source8')[0]
assert 'int wm=warp%4,wn=warp/4;' in source
assert 'unsigned c[4][Groups][2][4]={};' in source
assert 'x=__reduce_xor_sync(0xffffffffu,x);' in source
assert 'int idx=((wm*4+row)*(BN/16)+wn*Groups+group)*16+(rb&15u);' in source
owners={};tasks=[]
for w in range(12):
 wm,wn=w%4,w//4;tiles=[];slots=[]
 for row in range(4):
  for group in range(4):
   tile_row=wm*4+row;tile_col=wn*4+group
   tiles.append([tile_row,tile_col])
   for slot in range(16):
    index=(tile_row*12+tile_col)*16+slot
    assert index not in owners,(w,index,owners.get(index))
    owners[index]=w;slots.append(index)
 #32 rank128 folds atK4096:both writes to each transcript slot have same owner.
 for rb in range(32):
  for tr,tc in tiles:assert owners[(tr*12+tc)*16+(rb&15)]==w
 tasks.append(dict(producer_warp=w,consumer_warp=w,transcript_tiles=tiles,shared_words=slots,mma_region_rows=[wm*64,wm*64+63],mma_region_cols=[wn*64,wn*64+63],accumulator_int32_per_lane=128))
assert len(owners)==16*12*16==3072
assert set(owners)==set(range(3072))
lines=['Frozen fastest binary SHA256:'+v['sha256'],
'12producer warps ->12 transcript tasks. Each task contains16tiles;CTA total192tiles.',
'Each warp owns64x64 output region,128accumulator INT32 values/lane (4096/warp).',
'Transcript R128,K4096 has32fold boundaries;each tile has16stored state words,updated by its same warp on folds16..31.',
'warp producer->consumer CTA_rows CTA_cols transcript_tile_rows transcript_tile_cols owned_state_words']
for t in tasks:
 w=t['producer_warp'];wm,wn=w%4,w//4
 lines.append(f"{w:2} W{w}->T{w}(W{w}) {t['mma_region_rows']} {t['mma_region_cols']} [{wm*4},{wm*4+3}] [{wn*4},{wn*4+3}] 256")
lines+=['',
'Exact SASS dependency boundary:',
'0x16c0 cp.async wait:own thread pending copies,NOT other warp transcript completion.',
'0x16d0 nonblocking peer named arrival:scoreboard mask6 waits own final LDSM. It publishes to peers;does not wait for their arrivals.',
'0x16e0..17d0 remaining16MMA within same warp.',
'0x17e0 fold guard;0x1870 local XOR;0x1940/0x2320 REDUX paths. REDUX is within this warp.',
'0x2a00 own-group blocking named sync occurs AFTER this warp transcript stores.',
'0x2a50 final CTA barrier precedes cross-warp collective global output copy.',
'',
'Finding:current T_i arithmetic has no barrier dependency on a different producer warp W_j final accumulator/transcript result.',
'All3072shared transcript words have exactly one owning warp;producer=consumer for every tile and both write/update passes.',
'Cross-warp A/B readiness is a real operand dependency earlier in GEMM. Late handoff can indirectly delay the NEXT iteration and its future transcript,not the already-ready current T_i.',
'Adding a new per-T_i named barrier does not remove a current cross-warp dependency;it would add a rendezvous unless ownership or execution architecture changes.',
'Independent consumer warp T_i is a distinct redesign:register values are warp-private and require explicit transport. Not implemented or timed by this audit.',
'',
'Boundaries:static ownership/control-flow proof,not measurement of exact start cycles or scheduler utilization. No speedup claimed.',
'No GPU runs,register changes,barrier changes,production writes or live mining.',
'Source references:first namespace fold_defer_source8 lines128-132 and221-245;frozen SASS is authority for actual publication placement.']
(out/'ownership.json').write_text(json.dumps({'baseline':v['sha256'],'tasks':tasks,'unique_words':len(owners)},indent=2)+'\n')
(out/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
