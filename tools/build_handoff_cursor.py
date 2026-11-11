"""Hoisted copy origins vs cross-K offset induction on the validated lean handoff."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];s=(root/'artifacts/handoff-lean-source.cu').read_text()
def section(ns):
 a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);return s[a:b]
base=section('handoff_lean_baseline');control=section('handoff_lean_toggle')
helper='''
__device__ __forceinline__ void copy_cursor(const signed char* a,const signed char* b,
 signed char* sa,signed char* sb,unsigned offA,unsigned offB,int grB,int n) {
    unsigned dstA=__cvta_generic_to_shared(sa+threadIdx.x*16);
    unsigned dstB=__cvta_generic_to_shared(sb+threadIdx.x*16);
    #pragma unroll
    for(int j=0;j<3;++j) if(threadIdx.x+j*THREADS<BM*64/16) {
        const signed char* src=a+offA;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,16;" :: "r"(dstA),"l"(src));
        offA+=96u*4096u;dstA+=THREADS*16;
    }
    #pragma unroll
    for(int j=0;j<2;++j) {
        bool valid=grB<n;
        const signed char* src=b+(valid?offB:0u);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dstB),"l"(src),"r"(valid?16:0));
        grB+=96;offB+=96u*4096u;dstB+=THREADS*16;
    }
    asm volatile("cp.async.commit_group;");
}
'''
names=['baseline','control','hoist','induction'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for name in names:
 old=base if name=='baseline' else control;new=old
 if name in ('hoist','induction'):
  at=new.index('template<int Lead,int TileK=64');new=new[:at]+helper+new[at:]
  start=new.index('    #pragma unroll\n    for(int p=0;p<Stages-1;++p)');end=new.index('    static_assert(Stages==2',start)
  new=new[:start]+'''    static_assert(PackedB && !Planar && Layout==6 && TileK==64 && Stages==2 && !Wide);
    const unsigned logical=unsigned(threadIdx.x)^(unsigned(threadIdx.x>>3)&3u);
    const unsigned copy_row=logical/4u,copy_col=(logical%4u)*16u;
    const int copy_grB=cn*BN+int(copy_row);
    const unsigned originA=(unsigned(cm)*BM+copy_row)*4096u+copy_col;
    const unsigned originB=unsigned(copy_grB)*4096u+copy_col;
    copy_cursor(a,b,sa,sb,originA,originB,copy_grB,n);
'''+new[end:]
  call='            stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);';assert new.count(call)==1
  if name=='hoist':
   new=new.replace(call,'            copy_cursor(a,b,sa+next*BM*TileK,sb+next*BN*TileK,originA+unsigned((kt+1)*64),originB+unsigned((kt+1)*64),copy_grB,n);')
  else:
   point='    unsigned own_bar=4u+unsigned(threadIdx.x>=192);';assert new.count(point)==1
   new=new.replace(point,'    unsigned cursorA=originA+64u,cursorB=originB+64u;\n'+point)
   new=new.replace(call,'            copy_cursor(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cursorA,cursorB,copy_grB,n);\n        cursorA+=64u;cursorB+=64u;')
 ns='handoff_lean_baseline' if name=='baseline' else 'handoff_lean_toggle'
 (root/f'artifacts/handoff-cursor-{name}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='validated-'+ns,tofile=name)))
 parts.append(new.replace(ns,'handoff_cursor_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int handoff_cursor_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return handoff_cursor_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int handoff_cursor_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=handoff_cursor_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/handoff-cursor-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-handoff-cursor.so')],check=True)
