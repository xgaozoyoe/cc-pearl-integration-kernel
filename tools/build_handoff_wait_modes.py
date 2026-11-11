"""Source-only SM89 handoff controls; never replace the production library."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
source=(root/'artifacts/transcript-handoff-source.cu').read_text()
def section(ns):
 a=source.index('namespace '+ns);b=source.index('// namespace '+ns,a)+len('// namespace '+ns)
 return source[a:b]
base=section('transcript_handoff_baseline');spin=section('transcript_handoff_overlap')
names=['baseline','spin','token','delegate'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for name in names:
 old=base if name=='baseline' else spin;new=old
 if name=='token':
  new=new.replace('void handoff_arrive(unsigned bar)', 'unsigned long long handoff_arrive(unsigned bar)')
  new=new.replace('    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");', '    unsigned long long token;\n    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(token) : "r"(bar) : "memory");\n    return token;')
  new=new.replace('unsigned bar,unsigned phase)', 'unsigned bar,unsigned long long token)')
  new=new.replace('mbarrier.test_wait.parity.shared.b64', 'mbarrier.test_wait.shared.b64').replace('"r"(bar),"r"(phase)', '"r"(bar),"l"(token)')
  new=new.replace('        handoff_arrive(handoff_bar);', '        unsigned long long token=handoff_arrive(handoff_bar);').replace('handoff_wait(handoff_bar,unsigned(kt&1));','handoff_wait(handoff_bar,token);')
 if name=='delegate':
  new=new.replace('        handoff_wait(handoff_bar,unsigned(kt&1));', '''        // All 384 threads still arrive and publish cp.async completion.
        // Warp0 acquires that completion; CTA barrier propagates visibility
        // and prevents any next-iteration overwrite before all readers finish.
        if(threadIdx.x<32) handoff_wait(handoff_bar,unsigned(kt&1));
        __syncthreads();''')
 ns='transcript_handoff_baseline' if name=='baseline' else 'transcript_handoff_overlap'
 (root/f'artifacts/handoff-wait-modes-{name}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='validated-'+ns,tofile=name)))
 parts.append(new.replace(ns,'handoff_wait_modes_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int handoff_wait_modes_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return handoff_wait_modes_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int handoff_wait_modes_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=handoff_wait_modes_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/handoff-wait-modes-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-handoff-wait-modes.so')],check=True)
