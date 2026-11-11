"""SM89 symmetric two-group split named barriers; all warps retain MMA work.
Four named barriers alternate by K parity, preventing early arrival into a
peer's still-active previous phase. Production backend is never touched.
"""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
source=(root/'artifacts/transcript-handoff-source.cu').read_text()
def section(ns):
 a=source.index('namespace '+ns);b=source.index('// namespace '+ns,a)+len('// namespace '+ns)
 return source[a:b]
base=section('transcript_handoff_baseline');spin=section('transcript_handoff_overlap')
names=['baseline','spin','late','overlap'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for name in names:
 old=spin if name=='spin' else base;new=old
 if name in ('late','overlap'):
  start=new.index('        if constexpr(Stages==2) asm volatile("cp.async.wait_group 0;");');end=new.index('        int current=kt%Stages',start)
  new=new[:start]+'''        if(kt==0) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
        // Two groups of six warps, no dedicated producers. IDs1/2 for
        // even iterations,3/4 for odd; barrier0 remains initialization/output.
        const unsigned group=threadIdx.x/192;
        const unsigned own_bar=1+2*(kt&1)+group;
        const unsigned peer_bar=1+2*(kt&1)+(group^1);
'''+new[end:]
  arrive='''        // Publish this group's completed future copies and old-stage reads.
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        asm volatile("bar.arrive %0, 384;" :: "r"(peer_bar) : "memory");
'''
  point='        if constexpr(TileK==64) {\n        if((kt+1)%2==0) {';assert new.count(point)==1
  if name=='overlap':new=new.replace(point,arrive+point)
  point='    }\n    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;';assert new.count(point)==1
  wait='        asm volatile("bar.sync %0, 384;" :: "r"(own_bar) : "memory");\n'
  new=new.replace(point,(arrive if name=='late' else '')+wait+point)
 ns='transcript_handoff_overlap' if name=='spin' else 'transcript_handoff_baseline'
 (root/f'artifacts/handoff-named-{name}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='validated-'+ns,tofile=name)))
 parts.append(new.replace(ns,'handoff_named_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int handoff_named_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return handoff_named_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int handoff_named_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=handoff_named_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/handoff-named-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-handoff-named.so')],check=True)
