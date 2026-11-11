"""Isolated delayed fragment-fold experiment on the current A-stream kernel."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();ns='rank128_astream_candidate_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
s=s[a:b]
base=s.replace(ns,'rank128_fold_interleave_baseline');new=s.replace(ns,'rank128_fold_interleave_fast')
start=new.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)')
opening=new.index('{',start);depth=1;end=opening+1
while depth:
 depth+=(new[end]=='{')-(new[end]=='}');end+=1
new=new[:start]+new[end:]
point='''                        ar[row]=nextA;
                }
            }
            if constexpr(TileK!=64)'''
fold='''                        ar[row]=nextA;
                }
                // Previous output fragment is complete; current fragment's
                // two independent MMAs separate its last producer from XOR.
                if(slice==1 && (kt+1)%2==0 && group*4+row>0) {
                    constexpr int Dummy=0;
                    int previous_row=(row+3)%4;
                    int previous_group=group-(row==0);
                    unsigned x=0;
                    #pragma unroll
                    for(int h=0;h<2;++h) {
                        #pragma unroll
                        for(int v=0;v<4;++v)x^=c[previous_row][previous_group][h][v];
                    }
                    x=__reduce_xor_sync(0xffffffffu,x);
                    if(lane==0) {
                        int rb=(kt+1)/2-1;
                        int idx=((wm*4+previous_row)*(BN/16)+wn*Groups+previous_group)*16+rb%16;
                        if(rb<16)transcript[idx]=x;
                        else {unsigned prev=transcript[idx];transcript[idx]=((prev<<13)|(prev>>19))^x;}
                    }
                }
                if(slice==1 && (kt+1)%2==0 && group==Groups-1 && row==3) {
                    unsigned x=0;
                    #pragma unroll
                    for(int h=0;h<2;++h) {
                        #pragma unroll
                        for(int v=0;v<4;++v)x^=c[3][Groups-1][h][v];
                    }
                    x=__reduce_xor_sync(0xffffffffu,x);
                    if(lane==0) {
                        int rb=(kt+1)/2-1;
                        int idx=((wm*4+3)*(BN/16)+wn*Groups+Groups-1)*16+rb%16;
                        if(rb<16)transcript[idx]=x;
                        else {unsigned prev=transcript[idx];transcript[idx]=((prev<<13)|(prev>>19))^x;}
                    }
                }
            }
            if constexpr(TileK!=64)'''
assert new.count(point)==1
new=new.replace(point,fold).replace('                    constexpr int Dummy=0;\n','')
source='#include <cuda_runtime.h>\n#include <cstdint>\n'+base+'\n'+new+r'''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int fold_interleave_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(m%256||m>1048576||n>1048576)return cudaErrorInvalidValue;
 if(v==0)return rank128_fold_interleave_baseline::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 if(v==1)return rank128_fold_interleave_fast::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 return cudaErrorInvalidValue;
}
extern "C" int fold_interleave_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 auto e=fold_interleave_fold(v,a,b,m,n,t,nullptr);if(e)return e;
 launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''
path=root/'artifacts/rank128-fold-interleave-source.cu';path.write_text(source)
a=base.replace('rank128_fold_interleave_baseline','rank128_trial');b=new.replace('rank128_fold_interleave_fast','rank128_trial')
(root/'artifacts/rank128-fold-interleave-kernel.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='current-astream.cu',tofile='interleaved-fold.cu')))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),'-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-fold-interleave-experimental.so')],check=True)
