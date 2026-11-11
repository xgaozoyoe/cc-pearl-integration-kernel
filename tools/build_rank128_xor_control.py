"""Matched local-XOR instruction/partial-state controls. Never replace production."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-fold_local_late1-source.cu').read_text()
def extract(ns):
 a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
 return s[a:b]
base=extract('rank128_fold_local_late1_baseline').replace('rank128_fold_local_late1_baseline','xor_control_current')
early=extract('rank128_fold_local_late1_fast').replace('rank128_fold_local_late1_fast','xor_control_early')
late=early.replace('xor_control_early','xor_control_late')
a=late.index('                // One fragment of independent MMA')
b=late.index('            if constexpr(TileK!=64)',a)
late=late[:a]+'''            }
            // Same four partials, same PTX XOR, only evaluated after all MMA.
            if(slice==1 && (kt+1)%2==0 && group==Groups-1) {
                #pragma unroll
                for(int row=0;row<4;++row)
                    partial[row][0]=local_xor(c[row][group][0],c[row][group][1]);
            }
'''+late[b:]
wrapper=r'''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int xor_control_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(m%256||m>1048576||n>1048576)return cudaErrorInvalidValue;
 if(v==0)return xor_control_current::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 if(v==1)return xor_control_early::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 if(v==2)return xor_control_late::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 return cudaErrorInvalidValue;
}
extern "C" int xor_control_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 auto e=xor_control_fold(v,a,b,m,n,t,nullptr);if(e)return e;
 launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''
path=root/'artifacts/rank128-xor-control-source.cu';path.write_text('#include <cuda_runtime.h>\n#include <cstdint>\n'+base+'\n'+early+'\n'+late+wrapper)
(root/'artifacts/rank128-xor-control-kernel.diff').write_text(''.join(difflib.unified_diff(early.replace('xor_control_early','control').splitlines(True),late.replace('xor_control_late','control').splitlines(True),fromfile='early-local-xor.cu',tofile='late-local-xor.cu')))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),'-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-xor-control-experimental.so')],check=True)
