"""Offline region protocol overhead controls; never replaces production."""
from pathlib import Path
import subprocess, json, hashlib, re, difflib
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/region-compact';out.mkdir(exist_ok=True)
parent=root/'artifacts/region-free'
meta=json.loads((parent/'manifest.json').read_text())
manifest={'variants':{}}
for name in ('native','region_deferred'):
    v=meta['variants'][name];blob=(parent/f'{name}.cubin').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==v['sha256']
    (out/f'{name}.cubin').write_bytes(blob);manifest['variants'][name]=v
base=(parent/'region_deferred.cu').read_text()
compact=r'''__device__ __forceinline__ void rf_wait(unsigned bar,unsigned parity) {
 // Every participating warp is full and all lanes use the same region/phase.
 // All lanes still execute test_wait. Vote makes the loop exit uniform.
 asm volatile("{ .reg .pred p,all_done;\n"
              "region_poll:\n"
              "mbarrier.test_wait.parity.shared.b64 p,[%0],%1;\n"
              "vote.sync.all.pred all_done,p,0xffffffff;\n"
              "@!all_done bra.uni region_poll;\n}"
              :: "r"(bar),"r"(parity) : "memory");
}
'''
for name in ('compact_poll','hoist_only','compact_hoist'):
    s=base
    a=s.index('__device__ __forceinline__ void rf_wait(')
    b=s.index('__device__ __forceinline__ void rf_ready(',a)
    if name!='hoist_only': s=s[:a]+compact+s[b:]
    if name in ('hoist_only','compact_hoist'):
        s=s.replace('bool FixedCopy=false,int Layout=0>\n__device__ __forceinline__ void stage_copy',
                    'bool FixedCopy=false,int Layout=0,bool Reuse=false>\n__device__ __forceinline__ void stage_copy')
        needle='    int logical=int(threadIdx.x)^(int(threadIdx.x>>3)&3);'
        s=s.replace(needle, '    unsigned free_base=bars+16+56*(kt&1);\n    unsigned free_phase=unsigned((kt-2)/2)&1u;\n'+needle)
        for mat,off in [('false','0'),('true','32')]:
            old=f'region_wait(bars,kt,(threadIdx.x+j*THREADS)/4,{mat});'
            new=f'if constexpr(Reuse) rf_wait(free_base+{off}+8*((threadIdx.x+j*THREADS)/256),free_phase);'
            assert old in s;s=s.replace(old,new)
        old='            stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+1,m,n,k,bars);'
        assert old in s
        s=s.replace(old,'            if(kt>0)\n'+old.replace('Layout>','Layout,true>')+'\n            else\n'+old)
    s=s.replace('region_free_region_deferred','region_compact_'+name)
    path=out/f'{name}.cu';path.write_text(s)
    (out/f'{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='region_deferred',tofile=name)))
    p=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(path),'-o',str(out/f'{name}.cubin')],capture_output=True,text=True)
    (out/f'{name}.build.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
    sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True)
    (out/f'{name}.sass').write_text(sass)
    manifest['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/f'{name}.cubin').read_bytes()).hexdigest(),smem=69760,correct=True)
    print(name,p.stderr,flush=True)
manifest['variants']['compact_hoist']['correct']=False
manifest['variants']['compact_hoist']['safety_status']='Excluded: racecheck reports 5 read/write errors despite numerical PASS'
manifest['variants']['compact_poll']['correct']=False
manifest['variants']['compact_poll']['safety_status']='Excluded: spill and unvalidated polling change related to failed compact_hoist'
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
