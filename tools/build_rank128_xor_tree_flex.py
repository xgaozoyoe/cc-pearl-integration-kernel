"""Replace only eight-word lane XOR by a balanced 3+3+2 tree; isolated SM89."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();ns='rank128_astream_candidate_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);s=s[a:b]
base=s.replace(ns,'xor_tree_flex_original');new=s.replace(ns,'xor_tree_flex_balanced')
a=new.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)');tail=new[a:]
old='''                    unsigned x=0;
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int v=0;v<4;++v) x^=c[row][group][half][v];
                    }'''
replacement=r'''                    unsigned x;
                    // Keep three independent subtrees visible to the compiler;
                    // opaque final LOP3 prevents reassociation into one chain.
                    unsigned p=c[row][group][0][0]^c[row][group][0][1]^c[row][group][0][2];
                    unsigned q=c[row][group][0][3]^c[row][group][1][0]^c[row][group][1][1];
                    unsigned r=c[row][group][1][2]^c[row][group][1][3];
                    asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(x) : "r"(p),"r"(q),"r"(r));'''
assert tail.count(old)==1;tail=tail.replace(old,replacement);new=new[:a]+tail
wrapper=r'''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int xor_tree_flex_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(m%256||m>1048576||n>1048576)return cudaErrorInvalidValue;
 if(v==0)return xor_tree_flex_original::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 if(v==1)return xor_tree_flex_balanced::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 return cudaErrorInvalidValue;
}
extern "C" int xor_tree_flex_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 auto e=xor_tree_flex_fold(v,a,b,m,n,t,nullptr);if(e)return e;
 launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''
path=root/'artifacts/rank128-xor-tree-flex-source.cu';path.write_text('#include <cuda_runtime.h>\n#include <cstdint>\n'+base+'\n'+new+wrapper)
(root/'artifacts/rank128-xor-tree-flex-kernel.diff').write_text(''.join(difflib.unified_diff(base.replace('xor_tree_flex_original','control').splitlines(True),new.replace('xor_tree_flex_balanced','control').splitlines(True),fromfile='serial-lane-xor.cu',tofile='balanced-lane-xor.cu')))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),'-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-xor-tree-flex-experimental.so')],check=True)
