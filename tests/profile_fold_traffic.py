"""One chosen fold operator, 32 warmup launches then 3 measured launches."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys,ctypes
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
import cuda_capi as cc

lib=ctypes.CDLL(os.environ.get('PEARL_TRIAL_LIBRARY','/tmp/libpearl-fold-ops.so'))
P=ctypes.c_void_p;I=ctypes.c_int
lib.trial_fold.argtypes=[I,P,P,I,I,I,I,P]
lib.trial_info.argtypes=[I,P]
v=int(sys.argv[1]);m=n=k=4096;r=256
info=np.empty(5,np.int32)
cc._chk(lib.trial_info(v,info.ctypes.data_as(P)),'info')
rng=np.random.default_rng(100)
a=cc.DBuf(m*k);b=cc.DBuf(n*k);t=cc.DBuf((m//16)*(n//16)*64)
try:
    a.from_host(rng.integers(-128,128,(m,k),dtype=np.int8))
    b.from_host(rng.integers(-128,128,(n,k),dtype=np.int8))
    for _ in range(32):
        cc._chk(lib.trial_fold(v,a.ptr,b.ptr,m,n,k,r,t.ptr),'warmup')
    cc.sync()
    for _ in range(3):
        cc._chk(lib.trial_fold(v,a.ptr,b.ptr,m,n,k,r,t.ptr),'profile')
    cc.sync()
    print(f'variant={v}; updates={(m//16)*(n//16)*(k//r)}; resources={info.tolist()}',flush=True)
finally:
    for buf in (a,b,t):buf.free()
