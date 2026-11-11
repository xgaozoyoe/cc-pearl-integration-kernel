"""Randomized warp-level protocol model; no CUDA/performance simulation."""
from pathlib import Path
import random,json
W,K=12,64

def run(seed,late=False,omit=''):
 rng=random.Random(seed);queues=[]
 for w in range(W):
  q=[('copy',0)]
  for k in range(K):
   if k+1<K:
    if k and not late:q.append(('wait_free',k-1))
    q.append(('copy',k+1))
   if not late or k==0:q.append(('wait_ready',k))
   q.extend([('read',k),('free',k),('fold',k)])
   if late and k+1<K:q.extend([('wait_free',k),('wait_ready',k+1)])
  queues.append(q)
 pos=[0]*W;pending=[];written=[set() for _ in range(K)];read=[set() for _ in range(K)];free=[set() for _ in range(K)]
 observed_ready=set();observed_free=set();operations=0
 while any(pos[w]<len(queues[w]) for w in range(W)) or pending:
  eligible=[]
  for w in range(W):
   if pos[w]==len(queues[w]):continue
   op,k=queues[w][pos[w]]
   if op=='wait_ready' and omit!='ready' and len(written[k])<W:continue
   if op=='wait_free' and omit!='free' and len(free[k])<W:continue
   eligible.append(('warp',w))
  # Preserve each warp's asynchronous copy ordering.
  for i,(w,k) in enumerate(pending):
   if not any(ww==w for ww,kk in pending[:i]):eligible.append(('complete',i))
  assert eligible,'deadlock'
  kind,i=rng.choice(eligible);operations+=1
  if kind=='complete':
   w,k=pending.pop(i)
   assert k<2 or len(read[k-2])==W,('overwrite',w,k)
   assert k<2 or k-2 in observed_ready,('ready phase reused without observation',k)
   assert w not in written[k];written[k].add(w)
   continue
  w=i;op,k=queues[w][pos[w]];pos[w]+=1
  if op=='copy':pending.append((w,k))
  elif op=='wait_ready':observed_ready.add(k)
  elif op=='wait_free':observed_free.add(k)
  elif op=='read':
   assert len(written[k])==W,('unready read',w,k)
   assert k+2>=K or not written[k+2],('overwritten read',w,k)
   read[k].add(w)
  elif op=='free':
   assert w in read[k]
   assert k<2 or k-2 in observed_free,('free phase reused without observation',k)
   assert w not in free[k];free[k].add(w)
 assert all(len(x)==W for x in written+read+free)
 return operations

results={}
for late in (False,True):
 count=sum(run(seed,late) for seed in range(100))
 results['late' if late else 'deferred']={'schedules':100,'operations':count,'pass':True}
for omit in ('ready','free'):
 detected=False
 for seed in range(100):
  try:run(seed,False,omit)
  except AssertionError as e:detected=True;results['negative_'+omit]={'seed':seed,'detected':str(e)};break
 assert detected,omit
p=Path(__file__).resolve().parents[1]/'artifacts/ready-free/protocol-model.json';p.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
