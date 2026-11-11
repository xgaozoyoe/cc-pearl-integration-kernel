"""GPU0-only full-A-tile fast path R128 test. Production library is never replaced."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,ctypes as C,json,re,statistics,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
import cuda_capi as cc
import miner_capi as miner
import pool_common
P,I=C.c_void_p,C.c_int
lib=C.CDLL('/tmp/libpearl-rank128-astream1.so')
lib.astream1_fold.argtypes=[I,P,P,I,I,P,P]
lib.astream1_pow.argtypes=[I,P,P,I,I,P,P,P,P,P,P]
pack=C.CDLL(str(root/'artifacts/libpearl-rank128-astream-candidate.so'))
pack.rank_fold.argtypes=[I,P,P,I,I,P,P]
pack.rank_pow.argtypes=[I,P,P,I,I,P,P,P,P,P,P]
ref=C.CDLL(str(root/'artifacts/libpearl-rank.so'))
ref.rank_pow.argtypes=[I,P,P,I,I,P,P,P,P,P,P]
def pow_call(v,a,b,m,n,key,target,t,d=None,f=None,c=None):
 fn=pack.rank_pow if v==1 else lib.astream1_pow
 cc._chk(fn(128 if v==1 else v,cc._as(a),cc._as(b),m,n,cc._as(key),cc._as(target),cc._as(t),cc._as(d),cc._as(f),cc._as(c)),'astream1 pow')
p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate']);p.add_argument('--variant',type=int,default=0);p.add_argument('--small-only',action='store_true');args=p.parse_args()
resources={}
for v in (0,1):
 info=np.zeros(5,np.int32)
 fn=pack.rank_fold if v==1 else lib.astream1_fold
 cc._chk(fn(128 if v==1 else v,None,None,8192,8192,None,info.ctypes.data),'info')
 resources[v]=info.tolist()
print('resources',resources,flush=True)
if args.mode=='validate':
 results=[]
 for m,n in (((256,48),(256,384),(256,400)) if args.small_only else ((256,48),(256,384),(256,400),(8192,8192))):
  for seed in (42,1234):
   rng=np.random.default_rng(seed);alloc=[]
   def buf(size):
    x=cc.DBuf(size);alloc.append(x);return x
   a,b=buf(m*4096),buf(n*4096);a.from_host(rng.integers(-128,128,(m,4096),np.int8));b.from_host(rng.integers(-128,128,(n,4096),np.int8))
   key,target=buf(32),buf(32);key.from_host(rng.integers(0,256,32,np.uint8));target.memset(0)
   t,d=buf(m*n//256*64),buf(m*n//256*32)
   cc._chk(ref.rank_pow(128,a.ptr,b.ptr,m,n,key.ptr,target.ptr,t.ptr,d.ptr,None,None),'frozen')
   expected_t=np.empty(m*n//256*16,np.uint32);expected_d=np.empty(m*n//256*32,np.uint8)
   t.to_host(expected_t);d.to_host(expected_d)
   for v in (0,1):
    t.memset(0xa5);d.memset(0xa5);pow_call(v,a,b,m,n,key,target,t,d)
    actual_t=np.empty_like(expected_t);actual_d=np.empty_like(expected_d);t.to_host(actual_t);d.to_host(actual_d)
    assert np.array_equal(actual_t,expected_t),(v,m,n,seed,'transcript')
    assert np.array_equal(actual_d,expected_d),(v,m,n,seed,'digest')
    row=dict(variant=v,m=m,n=n,seed=seed,transcript_pass=True,digest_pass=True);results.append(row);print('PASS',row,flush=True)
   for x in alloc:x.free()
 Path('artifacts/rank128-astream-package-correctness-small.json' if args.small_only else 'artifacts/rank128-astream-package-correctness.json').write_text(json.dumps(results,indent=2)+'\n')
