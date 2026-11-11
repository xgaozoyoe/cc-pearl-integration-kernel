"""Validate all diagnostic sinks against CPU int32 accumulators, GPU0 only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import ctypes as C,sys,json
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
import cuda_capi as cc
lib=C.CDLL(str(root/'artifacts/libpearl-rank128-astream-ablation.so'))
lib.astream_ablation.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p]
ref=C.CDLL(str(root/'artifacts/libpearl-rank.so'));ref.rank_fold.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p]
m,n=256,384
# Dimensions: CTA,warp,row16,col16,lane,element8.
w=np.arange(12)[None,:,None,None,None,None]
r=np.arange(4)[None,None,:,None,None,None]
g=np.arange(4)[None,None,None,:,None,None]
l=np.arange(32)[None,None,None,None,:,None]
v=np.arange(8)[None,None,None,None,None,:]
ri=(w%4)*64+r*16+l//4+((v%4)//2)*8
ci=np.arange(2)[:,None,None,None,None,None]*192+(w//4)*64+g*16+(v//4)*8+2*(l%4)+(v%2)
results=[]
for seed in (42,1234):
 rng=np.random.default_rng(seed);a=rng.integers(-128,128,(m,4096),np.int8);b=rng.integers(-128,128,(n,4096),np.int8)
 acc=np.zeros((m,n),np.int32);local=np.zeros((2,12,32),np.uint32);redux=local.copy()
 for chunk in range(32):
  sl=slice(chunk*128,(chunk+1)*128)
  acc+=a[:,sl].astype(np.int32)@b[:,sl].astype(np.int32).T
  vals=acc[ri,ci].view(np.uint32);x=np.bitwise_xor.reduce(vals,axis=-1)
  for row in range(4):
   for group in range(4):
    local=((local<<1)|(local>>31))^x[:,:,row,group,:]
    folded=np.bitwise_xor.reduce(x[:,:,row,group,:],axis=-1)[...,None]
    redux=((redux<<1)|(redux>>31))^folded
 final=np.bitwise_xor.reduce(np.bitwise_xor.reduce(x,axis=2),axis=2)
 expected={1:final.ravel(),2:(final^local).ravel(),3:(final^redux).ravel()}
 da,db,t=cc.DBuf(a.nbytes),cc.DBuf(b.nbytes),cc.DBuf(m*n//256*64);da.from_host(a);db.from_host(b)
 cc._chk(ref.rank_fold(128,da.ptr,db.ptr,m,n,t.ptr,None),'reference')
 et=np.empty(m*n//256*16,np.uint32);t.to_host(et)
 expected[0]=expected[4]=et
 for variant in range(5):
  t.memset(0xa5);cc._chk(lib.astream_ablation(variant,da.ptr,db.ptr,m,n,t.ptr,None),'test')
  out=np.empty_like(expected[variant]);t.to_host(out)
  assert np.array_equal(out,expected[variant]),(variant,seed,np.flatnonzero(out!=expected[variant])[:8])
  row=dict(variant=variant,seed=seed,pass_bit_exact=True);results.append(row);print(row,flush=True)
 for buf in (da,db,t):buf.free()
(root/'artifacts/rank128-astream-ablation-correctness.json').write_text(json.dumps(results,indent=2)+'\n')
