"""CPU mathematical checks: R128 fold, carry counterexample, output partitioning.
No GPU kernel, production library, target, proof or mining path is modified.
"""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MASK=(1<<32)-1
R=128;K=4096

def fold(xs):
 t=[0]*16
 for p,x in enumerate(xs):
  q=p%16
  if p<16:t[q]=int(x)
  else:t[q]=(((t[q]<<13)|(t[q]>>19))&MASK)^int(x)
 return t

def stream(a,bt):
 c=np.zeros((a.shape[0],bt.shape[0]),np.int64);xs=[]
 for start in range(0,K,R):
  c+=a[:,start:start+R].astype(np.int64)@bt[:,start:start+R].astype(np.int64).T
  xs.append(int(np.bitwise_xor.reduce((c.reshape(-1)&MASK).astype(np.uint32))))
 return fold(xs),xs

# Same XOR state, same next chunk increment, different required next XOR.
x0=[0,0];x1=[1,1];delta=[1,0]
assert (x0[0]^x0[1])==(x1[0]^x1[1])==0
next0=(x0[0]+delta[0])^(x0[1]+delta[1])
next1=(x1[0]+delta[0])^(x1[1]+delta[1])
assert next0==1 and next1==3
rows=[]
for case in ('seed42','seed1234','all_negative128','mixed_extremes'):
 if case.startswith('seed'):
  rng=np.random.default_rng(int(case[4:]));a=rng.integers(-128,128,(16,K),np.int16);b=rng.integers(-128,128,(16,K),np.int16)
 elif case=='all_negative128':a=np.full((16,K),-128,np.int16);b=a.copy()
 else:
  rng=np.random.default_rng(9);a=rng.choice(np.array([-128,127],np.int16),(16,K));b=rng.choice(np.array([-128,127],np.int16),(16,K))
 expected,xs=stream(a,b)
 # Only bit0 admits this simple bilinear parity factorization.
 pa=np.bitwise_xor.reduce(a.astype(np.int64)&1,axis=0)
 pb=np.bitwise_xor.reduce(b.astype(np.int64)&1,axis=0)
 for p,x in enumerate(xs):assert int(np.bitwise_xor.reduce((pa&pb)[:(p+1)*R]))==(x&1)
 for tm,tn in ((16,8),(8,16),(4,4),(1,1)):
  got=[0]*16
  for i in range(0,16,tm):
   for j in range(0,16,tn):
    part,_=stream(a[i:i+tm],b[j:j+tn]);got=[x^y for x,y in zip(got,part)]
  assert got==expected,(case,tm,tn)
  rows.append(dict(case=case,output_tile=[tm,tn],max_live_scalar_accumulators=tm*tn,transcript_exact=True))
result=dict(rank=R,k=K,updates=K//R,transcript_words=16,
 counterexample=dict(previous_vectors=[x0,x1],same_previous_xor=0,next_delta=delta,next_xors=[next0,next1]),
 partition_checks=rows,lsb_factorization_pass=True,
 note='CPU algebra checks only, no performance claim. Partitioning reduces live accumulators but changes operand reuse/data movement. No general lower bound on all possible Pearl algorithms is proved.')
(ROOT/'artifacts/transcript-state-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: carry counterexample;16 exact partition/fold comparisons;128 bit0 checks; no GPU used.')
