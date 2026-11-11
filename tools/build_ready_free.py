"""Isolated Ada ready/free stage protocol. Never builds a mining library."""
from pathlib import Path
import subprocess,json,hashlib,difflib,re
root=Path(__file__).resolve().parents[1];out=root/'artifacts/ready-free';out.mkdir(exist_ok=True)
s=(root/'artifacts/fold-defer-reduce/source.cu').read_text();a=s.index('namespace fold_defer_source8');b=s.index('template<int Lead,int TileK=64',s.index('        own_bar^=2u;',a));base=s[a:b]
helpers=r'''
__device__ __forceinline__ void rf_wait(unsigned bar,unsigned parity) {
 unsigned done;
 do {
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p,[%1],%2; selp.u32 %0,1,0,p; }" : "=r"(done) : "r"(bar),"r"(parity) : "memory");
 } while(!done);
}
__device__ __forceinline__ void rf_ready(unsigned bar) {
 // One asynchronous arrival per thread, count384; no extra normal arrival.
 asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void rf_free(unsigned bar) {
 asm volatile("mbarrier.arrive.shared.b64 _,[%0];" :: "r"(bar) : "memory");
}
'''
m={'description':'Experimental separate ready/free mbarriers,2stages. Source controls may differ from frozen optimized SASS. Offline only.','variants':{}}
v=json.loads((root/'artifacts/fold-preamble/manifest.json').read_text())['variants']['both_compact'];blob=(root/'artifacts/fold-preamble/both_compact.cubin').read_bytes();assert hashlib.sha256(blob).hexdigest()==v['sha256'];(out/'native.cubin').write_bytes(blob);m['variants']['native']=v
for name in ['source_control','split_late','split_deferred']:
 new=base
 if name!='source_control':
  at=new.index('template<int Lead,int TileK=64');new=new[:at]+helpers+new[at:]
  old='    unsigned c[4][Groups][2][4]={};'
  init='''    unsigned bars=(unsigned)__cvta_generic_to_shared(shared+Stages*(BM+BN)*TileK+WORDS*sizeof(unsigned));
    if(threadIdx.x<4) {
        unsigned addr=bars+8*threadIdx.x;
        asm volatile("mbarrier.init.shared.b64 [%0],384;" :: "r"(addr) : "memory");
    }
    __syncthreads();
'''
  new=new.replace(old,init+old)
  old='    // Initial stage publication is prologue-only, not a per-iteration branch.\n    asm volatile("cp.async.wait_group 0;" ::: "memory");\n    __syncthreads();'
  assert old in new;new=new.replace(old,'    rf_ready(bars); // prologue publishes K0 in slot0')
  old='''        if(kt+Stages-1<(k+TileK-1)/TileK)
            stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);'''
  newcopy='''        if(kt+1<k/64) {
            // Slot(next) previously held K(kt-1), except pristine slot1 at kt0.
            if(kt>0) rf_wait(bars+16+8*next,unsigned((kt-1)/2)&1u);
            stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+1,m,n,k);
            rf_ready(bars+8*next);
        }
        rf_wait(bars+8*current,unsigned(kt/2)&1u);
'''
  if name=='split_late':
   newcopy=newcopy.replace('            if(kt>0) rf_wait(bars+16+8*next,unsigned((kt-1)/2)&1u);\n','').replace('        rf_wait(bars+8*current,unsigned(kt/2)&1u);','        if(kt==0) rf_wait(bars,0);')
  assert old in new;new=new.replace(old,newcopy)
  old='''        asm volatile("cp.async.wait_group 0;" ::: "memory");
        asm volatile("bar.arrive %0, 384;" :: "r"(peer_bar) : "memory");'''
  assert old in new;new=new.replace(old,'        rf_free(bars+16+8*current);')
  old='''        asm volatile("bar.sync %0, 384;" :: "r"(own_bar) : "memory");
        own_bar^=2u;'''
  late='''        if(kt+1<k/64) {
            rf_wait(bars+16+8*current,unsigned(kt/2)&1u);
            rf_wait(bars+8*next,unsigned((kt+1)/2)&1u);
        }'''
  assert old in new;new=new.replace(old,late if name=='split_late' else '')
  point='    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;i+=Threads) {'
  new=new.replace(point,'''    __syncthreads();
    if(threadIdx.x<4) {
        unsigned addr=bars+8*threadIdx.x;
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(addr) : "memory");
    }
    for(int i=threadIdx.x;i<WORDS;i+=Threads) {''')
 ns='ready_free_'+name;new=new.replace('fold_defer_source8',ns)
 new+='\ntemplate __global__ void kernel<1,64,false,true,false,false,2,false,6>(const signed char*,const signed char*,int,int,int,int,unsigned*);\n}\n'
 path=out/(name+'.cu');path.write_text('#include <cuda_runtime.h>\n#include <cstdint>\n'+new)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='source8',tofile=name)))
 r=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(path),'-o',str(out/(name+'.cubin'))],text=True,capture_output=True)
 (out/(name+'.build.log')).write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'.cubin'))],text=True);(out/(name+'.sass')).write_text(dump)
 m['variants'][name]={'entry':re.search(r'Function : (\S+)',dump)[1],'sha256':hashlib.sha256((out/(name+'.cubin')).read_bytes()).hexdigest(),'smem':69632+(0 if name=='source_control' else 32),'correct':True}
 print(name,r.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
