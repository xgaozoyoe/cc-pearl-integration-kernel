"""GPU0-only R128 experiment; never connects to a pool or changes production."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import statistics
import sys
import numpy as np
import blake3
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'python'))
import cuda_capi as cc
import miner_capi as miner
import pool_common
import pearl_host
import pearl_mining as pm

lib = C.CDLL('/tmp/libpearl-rank.so')
I, P = C.c_int, C.c_void_p
lib.rank_fold.argtypes = [I,P,P,I,I,P,P]
lib.rank_reference.argtypes = [I,P,P,I,I,P]
lib.rank_pow.argtypes = [I,P,P,I,I,P,P,P,P,P,P]

def pow_call(a,b,m,n,k,r,key,target,t,dig,found,coord,variant=0):
    assert k == 4096
    cc._chk(lib.rank_pow(r,cc._as(a),cc._as(b),m,n,cc._as(key),cc._as(target),
        cc._as(t),cc._as(dig),cc._as(found),cc._as(coord)), 'rank_pow')

def validate():
    results = []
    for rank in (256,128):
        info = np.zeros(5,np.int32)
        cc._chk(lib.rank_fold(rank,None,None,64,64,None,info.ctypes.data),'info')
        print('resources',rank,info.tolist(),flush=True)
        for seed in (42,1234):
            for m,n in ((32,48),(256,384)):
                rng = np.random.default_rng(seed)
                a = rng.integers(-128,128,(m,4096),np.int8)
                b = rng.integers(-128,128,(n,4096),np.int8)
                alloc=[]
                def buf(size):
                    x=cc.DBuf(size);alloc.append(x);return x
                da,db=buf(a.nbytes),buf(b.nbytes)
                da.from_host(a);db.from_host(b)
                t=buf(m*n//256*64);d=buf(m*n//256*32);key=buf(32);target=buf(32)
                key_bytes=bytes(range(32));key.from_host(np.frombuffer(key_bytes,np.uint8));target.memset(0)
                t.memset(0xa5)
                pow_call(da,db,m,n,4096,rank,key,target,t,d,None,None)
                actual=np.empty((m//16,n//16,16),np.uint32);digest=np.empty((m//16,n//16,32),np.uint8)
                t.to_host(actual);d.to_host(digest)
                expected=np.zeros_like(actual);acc=np.zeros((m,n),np.int32)
                for i in range(4096//rank):
                    acc += a[:,i*rank:(i+1)*rank].astype(np.int32) @ b[:,i*rank:(i+1)*rank].astype(np.int32).T
                    fold=np.bitwise_xor.reduce(np.bitwise_xor.reduce(acc.view(np.uint32).reshape(m//16,16,n//16,16),axis=3),axis=1)
                    prev=expected[:,:,i%16]
                    expected[:,:,i%16]=((prev<<13)|(prev>>19))^fold
                assert np.array_equal(expected,actual),(rank,seed,m,n,'CPU transcript')
                for row in range(m//16):
                    for col in range(n//16):
                        assert digest[row,col].tobytes()==blake3.blake3(expected[row,col].tobytes(),key=key_bytes).digest()
                if m%64==0 and n%64==0:
                    cc._chk(lib.rank_reference(rank,da.ptr,db.ptr,m,n,t.ptr),'DP4A')
                    t.to_host(actual)
                    assert np.array_equal(actual,expected),'DP4A transcript'
                results.append(dict(rank=rank,seed=seed,m=m,n=n,pass_cpu=True,pass_digest=True))
                print('PASS',results[-1],flush=True)
                for x in alloc:x.free()
    Path('artifacts/ada-rank-correctness.json').write_text(json.dumps(results,indent=2)+'\n')

def bench():
    cc.pearl_pow_split=pow_call
    miner.PROOF_PROCESS=False # target zero; no hits or proof snapshots
    class Pool:
        def check_newer_job(self,job):return None
    class Scheduler:
        mode='user'
        def __init__(self):self.grids=0
        def note(self,elapsed):pass
        def maybe_switch(self):
            self.grids+=1;return self.grids>=44
    original_setup=cc.setup_job
    results=[]
    for repeat in range(2):
        for rank in ((256,128) if repeat==0 else (128,256)):
            miner.R=pool_common.R=rank
            scheduler=Scheduler();times=[]
            cc.setup_job=lambda *a:original_setup(*a[:-1],202610070000+scheduler.grids)
            bufs=miner.Bufs(8192)
            def log(line):
                match=re.search(r'grid wall: ([\d.]+) s',line)
                if match:times.append(float(match[1])*1000)
            miner.mine_job(Pool(),pool_common.real_config(),bytes(80),0,'offline',8192,0,
                           scheduler,bufs,dict(user=0,dev=0),None,log)
            values=times[4:]
            raw=2**46/1e12/(statistics.mean(values)/1000)
            row=dict(rank=rank,repeat=repeat,samples=len(values),median_ms=statistics.median(values),
                     mean_ms=statistics.mean(values),p95_ms=float(np.percentile(values,95)),
                     raw_ths=raw,consensus_equivalent_ths=raw*128/rank)
            results.append(row);print(json.dumps(row),flush=True)
            for value in vars(bufs).values():
                if isinstance(value,cc.DBuf):value.free()
                elif isinstance(value,list):
                    for item in value:
                        if isinstance(item,cc.DBuf):item.free()
    Path('artifacts/ada-rank-results.json').write_text(json.dumps(results,indent=2)+'\n')

def proof_validation():
    # End-to-end setup/noise/search/proof with official salted V3 verifier.
    m,n,k,rank=128,256,4096,128
    pool_common.R=rank
    cfg=pool_common.real_config()
    nbits=0x1f000080
    hdr=pm.IncompleteBlockHeader(version=0,prev_block=bytes(32),merkle_root=bytes(range(32)),
                                 timestamp=0x66666666,nbits=nbits)
    key_bytes=pearl_host.derive_key(hdr.to_bytes(),cfg)
    bound=pm.penalized_target_bound((nbits&0xffffff) << (8*((nbits>>24)-3)),cfg)
    sizes=dict(a=m*k,b=k*n,bt=n*k,key=32,sa=32,sb=32,target=32,
               al=m*rank,ar=rank*k,bl=k*rank,br=n*rank,art=k*rank,
               an=m*k,bn=n*k,t=m*n//256*64,d=m*n//256*32,found=4,coord=8)
    d={name:cc.DBuf(size) for name,size in sizes.items()}
    try:
        d['key'].from_host(np.frombuffer(key_bytes,np.uint8))
        d['target'].from_host(np.frombuffer(bound.to_bytes(32,'little'),np.uint8))
        for seed in range(10):
            cc.setup_job(d['a'],d['b'],d['bt'],d['key'],d['sa'],d['sb'],m,n,k,rank,seed+42)
            a=np.empty((m,k),np.int8);b=np.empty((n,k),np.int8)
            d['a'].to_host(a);d['bt'].to_host(b)
            seeds=[]
            for name in ('sa','sb'):
                value=np.empty(32,np.uint8);d[name].to_host(value);seeds.append(value.tobytes())
            assert tuple(seeds)==pearl_host.commitment_hashes_bt_v3(a,b,key_bytes),'V3 seeds'
            cc.noise_gen(d['al'],d['ar'],d['bl'],d['br'],d['sa'],d['sb'],m,n,k,rank)
            cc.transpose_i8(d['ar'],d['art'],rank,k,k,0)
            cc.noise_gemm(d['al'],d['art'],d['a'],d['an'],m,k,rank)
            cc.noise_gemm(d['br'],d['bl'],d['bt'],d['bn'],n,k,rank)
            d['found'].memset(0)
            pow_call(d['an'],d['bn'],m,n,k,rank,d['sa'],d['target'],d['t'],d['d'],d['found'],d['coord'])
            found=np.zeros(1,np.int32);d['found'].to_host(found)
            if found[0]:
                coord=np.zeros(2,np.int32);d['coord'].to_host(coord)
                proof=pearl_host.build_proof_bt(a,b,int(coord[0]),int(coord[1]),key_bytes,rank)
                valid,message=pm.verify_plain_proof_v3(hdr,proof)
                assert valid,message
                result=dict(rank=rank,seed=seed+42,coord=coord.tolist(),v3_seed_pass=True,
                            official_v3_proof_pass=bool(valid),message=str(message))
                print(json.dumps(result),flush=True)
                Path('artifacts/ada-rank-proof.json').write_text(json.dumps(result,indent=2)+'\n')
                return
        raise AssertionError('no winning tile in 10 test grids')
    finally:
        for buf in d.values():buf.free()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','bench','proof']);args=p.parse_args()
    {'validate':validate,'bench':bench,'proof':proof_validation}[args.mode]()
