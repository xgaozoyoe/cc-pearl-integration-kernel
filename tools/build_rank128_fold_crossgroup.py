"""Delay boundary fold into the following K64 iteration, before each C update."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();ns='rank128_astream_candidate_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);s=s[a:b]
base=s.replace(ns,'rank128_fold_crossgroup_baseline');new=s.replace(ns,'rank128_fold_crossgroup_fast')
# Preserve the final K4096 boundary: there is no following iteration to host it.
needle='if((kt+1)%2==0) {';assert new.count(needle)==1
new=new.replace(needle,'if(kt+1==k/TileK) {')
needle='''                    mma<PackedB>(c[row][group][half],ar[row],br[slot],half);'''
fold='''                    // Consume the PREVIOUS boundary before this fragment is
                    // overwritten. Other fragments can already advance in K.
                    if(slice==0 && kt>0 && kt%2==0) {
                        unsigned x=0;
                        #pragma unroll
                        for(int h=0;h<2;++h) {
                            #pragma unroll
                            for(int v=0;v<4;++v)x^=c[row][group][h][v];
                        }
                        x=__reduce_xor_sync(0xffffffffu,x);
                        if(lane==0) {
                            int rb=kt/2-1;
                            int idx=((wm*4+row)*(BN/16)+wn*Groups+group)*16+rb%16;
                            if(rb<16)transcript[idx]=x;
                            else {unsigned prev=transcript[idx];transcript[idx]=((prev<<13)|(prev>>19))^x;}
                        }
                    }
'''
fold=fold.replace('if(slice==0 && kt>0 && kt%2==0) {', 'if(slice==0 && kt>0 && kt%2==0) {\n                        #pragma unroll\n                        for(int row=0;row<4;++row) {')
fold+='                    }\n'
point='            if constexpr(TileK!=64) { if(kt*TileK+slice*32>=k) continue; }'
assert new.count(point)==1;new=new.replace(point,point+'\n'+fold)
source='#include <cuda_runtime.h>\n#include <cstdint>\n'+base+'\n'+new+r'''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int fold_crossgroup_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(m%256||m>1048576||n>1048576)return cudaErrorInvalidValue;
 if(v==0)return rank128_fold_crossgroup_baseline::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 if(v==1)return rank128_fold_crossgroup_fast::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 return cudaErrorInvalidValue;
}
extern "C" int fold_crossgroup_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 auto e=fold_crossgroup_fold(v,a,b,m,n,t,nullptr);if(e)return e;
 launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''
path=root/'artifacts/rank128-fold-crossgroup-source.cu';path.write_text(source)
a=base.replace('rank128_fold_crossgroup_baseline','rank128_trial');b=new.replace('rank128_fold_crossgroup_fast','rank128_trial')
(root/'artifacts/rank128-fold-crossgroup-kernel.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='current-astream.cu',tofile='cross-iteration-fold.cu')))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),'-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-fold-crossgroup-experimental.so')],check=True)
