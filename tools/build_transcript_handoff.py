"""Correct-arithmetic SM89 split arrival/wait experiment. Isolated backend."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];ns='rank128_astream_candidate_fast'
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n'];names=['baseline','late','overlap']
helpers='''
__device__ __forceinline__ void handoff_arrive(unsigned bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(bar) : "memory");
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void handoff_wait(unsigned bar,unsigned phase) {
    unsigned done;
    do {
        asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(bar),"r"(phase) : "memory");
    } while(!done);
}
'''
for mode,name in enumerate(names):
 new=s
 if mode:
  at=new.index('template<int Lead,int TileK=64');new=new[:at]+helpers+new[at:]
  point='    unsigned c[4][Groups][2][4]={};'
  init='''    static_assert(Stages==2 && TileK==64 && !Wide);
    unsigned handoff_bar=(unsigned)__cvta_generic_to_shared(shared+Stages*(BM+BN)*TileK+WORDS*sizeof(unsigned));
    if(threadIdx.x==0) asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(handoff_bar), "r"(2*Threads) : "memory");
'''
  assert new.count(point)==1;new=new.replace(point,init+point)
  start=new.index('        if constexpr(Stages==2) asm volatile("cp.async.wait_group 0;");');end=new.index('        int current=kt%Stages',start)
  new=new[:start]+'''        if(kt==0) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads(); // also publishes mbarrier initialization
        }
'''+new[end:]
  point='        if constexpr(TileK==64) {\n        if((kt+1)%2==0) {';assert new.count(point)==1
  if mode==2:new=new.replace(point,'        handoff_arrive(handoff_bar);\n'+point)
  point='    }\n    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;';assert new.count(point)==1
  endcode=('        handoff_arrive(handoff_bar);\n' if mode==1 else '')+'        handoff_wait(handoff_bar,unsigned(kt&1));\n'
  new=new.replace(point,endcode+'    }\n    __syncthreads();\n    if(threadIdx.x==0) asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(handoff_bar) : "memory");\n    for(int i=threadIdx.x;i<WORDS;')
  new=new.replace('constexpr int smem=Stages*(BM+BN)*TileK+WORDS*sizeof(unsigned);','constexpr int smem=Stages*(BM+BN)*TileK+WORDS*sizeof(unsigned)+8;')
 (root/f'artifacts/transcript-handoff-{name}.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='current-baseline',tofile=name)))
 parts.append(new.replace(ns,'transcript_handoff_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int transcript_handoff_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return transcript_handoff_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int transcript_handoff_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=transcript_handoff_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/transcript-handoff-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-transcript-handoff.so')],check=True)
