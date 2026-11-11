"""Independent byte-layout oracle for isolated GPU0 prepack implementation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import sys,ctypes as C,json
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'tools'),str(root/'python')]
from check_gpu0_idle import check_gpu0_idle
check_gpu0_idle()
import cuda_capi as cc
out=root/'artifacts/prepack-k64';lib=C.CDLL(str(out/'libprepack.so'));lib.pack_input.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int]
results=[]
for height in (192,256):
 for rows in (16,192,256,400,800,8192):
  rng=np.random.default_rng(42+rows);x=rng.integers(-128,128,(rows,4096),np.int8);padded=((rows+height-1)//height)*height
  a,b=cc.DBuf(x.nbytes),cc.DBuf(padded*4096);a.from_host(x);b.memset(0xa5)
  cc._chk(lib.pack_input(a.ptr,b.ptr,rows,height),'pack')
  got=np.empty(padded*4096,np.int8);b.to_host(got)
  xp=np.zeros((padded,4096),np.int8);xp[:rows]=x
  # Logical stages first; independently permute their 16-byte chunks using byte swizzle.
  stages=xp.reshape(padded//height,height,64,64).transpose(0,2,1,3).copy().reshape(-1,height*4,16)
  byte=np.arange(height*4,dtype=np.uint32)*16
  source_chunk=(byte^((byte>>3)&0x30))//16
  expected=stages[:,source_chunk,:].reshape(-1)
  assert np.array_equal(got,expected),(rows,height)
  results.append(dict(rows=rows,height=height,byte_exact=True,zero_padding=True));a.free();b.free()
(out/'layout-validation.json').write_text(json.dumps(results,indent=2)+'\n');print('PASS',len(results),'independent byte-layout comparisons')
