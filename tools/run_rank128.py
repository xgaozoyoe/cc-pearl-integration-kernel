"""Explicit single-GPU R128 trial using an isolated, validated Ada library."""
import argparse
import ctypes as C
import hashlib
import os
from pathlib import Path
import sys

def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--wallet',required=True)
    p.add_argument('--worker',default='ada-rank128-test')
    p.add_argument('--pool',default='pearl-eu2.luckypool.io:3360')
    p.add_argument('--region',type=int,choices=[4096,8192],default=8192)
    p.add_argument('--library',type=Path,default=root/'artifacts/libpearl-rank.so')
    p.add_argument('--max-jobs',type=int,default=0)
    args=p.parse_args()
    if args.gpu<0:p.error('--gpu must select one nonnegative physical GPU index')
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    sys.path.insert(0,str(root/'python'))
    import cuda_capi as cc
    import miner_capi as miner
    import pool_common
    library=args.library.resolve()
    lib=C.CDLL(str(library))
    lib.rank_pow.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int]+[C.c_void_p]*6
    def pow_call(a,b,m,n,k,r,key,target,t,dig,found,coord,variant):
        if k!=4096 or r!=128:raise ValueError('R128 trial requires K4096/R128')
        cc._chk(lib.rank_pow(r,cc._as(a),cc._as(b),m,n,cc._as(key),cc._as(target),
                            cc._as(t),cc._as(dig),cc._as(found),cc._as(coord)),'rank128')
    cc.pearl_pow_split=pow_call
    miner.R=pool_common.R=128
    print(f'RANK128 TRIAL GPU={args.gpu} region={args.region} library={library} '
          f'sha256={hashlib.sha256(library.read_bytes()).hexdigest()}',flush=True)
    sys.argv=[sys.argv[0],'--single','--wallet',args.wallet,'--worker',args.worker,
              '--pool',args.pool,'--region',str(args.region),'--max-jobs',str(args.max_jobs)]
    miner.main()

if __name__=='__main__':
    main()
