"""Isolated source controls for named-barrier loop overhead; SM89 only."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/handoff-named-source.cu').read_text()
def section(ns):
 a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);return s[a:b]
base=section('handoff_named_baseline');named=section('handoff_named_overlap')
names=['baseline','named','peeled','toggle'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for name in names:
 old=base if name=='baseline' else named;new=old
 if name in ('peeled','toggle'):
  first='''        if(kt==0) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
'''
  assert new.count(first)==1;new=new.replace(first,'')
  loop='    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {';assert new.count(loop)==1
  new=new.replace(loop,'''    static_assert(Stages==2 && TileK==64 && !Wide);
    // Initial stage publication is prologue-only, not a per-iteration branch.
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
'''+loop)
 if name=='toggle':
  oldids='''        // Two groups of six warps, no dedicated producers. IDs1/2 for
        // even iterations,3/4 for odd; barrier0 remains initialization/output.
        const unsigned group=threadIdx.x/192;
        const unsigned own_bar=1+2*(kt&1)+group;
        const unsigned peer_bar=1+2*(kt&1)+(group^1);
'''
  assert new.count(oldids)==1
  new=new.replace(oldids,'''        // IDs4/5 and6/7 alternate. Same two-group protocol; no division,
        // parity shift or add is needed to select the barriers each iteration.
        const unsigned peer_bar=own_bar^1u;
''')
  loop='    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {'
  new=new.replace(loop,'    unsigned own_bar=4u+unsigned(threadIdx.x>=192);\n'+loop)
  wait='        asm volatile("bar.sync %0, 384;" :: "r"(own_bar) : "memory");';assert new.count(wait)==1
  new=new.replace(wait,wait+'\n        own_bar^=2u;')
 ns='handoff_named_baseline' if name=='baseline' else 'handoff_named_overlap'
 (root/f'artifacts/handoff-lean-{name}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='validated-'+ns,tofile=name)))
 parts.append(new.replace(ns,'handoff_lean_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int handoff_lean_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return handoff_lean_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int handoff_lean_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=handoff_lean_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/handoff-lean-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-handoff-lean.so')],check=True)
