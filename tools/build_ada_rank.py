"""Build isolated R128/R256 v121 kernels; never replace production library."""
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
source = (root / 'csrc/gemm/pearl-gemm-ada-v2.cu').read_text()
source = source[:source.index('cudaError_t launch_pearl_gemm_ada_v2')]
parts = []
for rank in (256, 128):
    s = source.replace('pearl_ada_v2', f'pearl_rank{rank}')
    if rank == 128:
        s = s.replace('r!=256', 'r!=128')
        s = s.replace('(kt+1)%4==0', '(kt+1)%2==0').replace('(kt+1)/4-1', '(kt+1)/2-1')
        s = s.replace('(kt*TileK+(slice+1)*32)%256==0', '(kt*TileK+(slice+1)*32)%128==0')
        s = s.replace('(kt*TileK+(slice+1)*32)/256-1', '(kt*TileK+(slice+1)*32)/128-1')
        old = 'if(lane==0) transcript[((wm*4+row)*(BN/16)+wn*Groups+group)*16+rb]=x;'
        assert s.count(old) == 2
        s = s.replace(old, '''if(lane==0) {
                        int idx=((wm*4+row)*(BN/16)+wn*Groups+group)*16+(rb%16);
                        if(rb<16) transcript[idx]=x;
                        else { unsigned prev=transcript[idx]; transcript[idx]=((prev<<13)|(prev>>19))^x; }
                    }''')
    parts.append(s)
parts.append('''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
void launch_pearl_gemm_only(const int8_t*,const int8_t*,int,int,int,int,uint32_t*,int,cudaStream_t);
extern "C" int rank_fold(int r,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    if(r==128)return pearl_rank128::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,r,t,info,nullptr);
    if(r==256)return pearl_rank256::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,r,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int rank_reference(int r,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t) {
    launch_pearl_gemm_only(a,b,m,n,4096,r,t,1,nullptr);
    return cudaGetLastError();
}
extern "C" int rank_pow(int r,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=rank_fold(r,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
''')
generated = Path('/tmp/pearl-rank-generated.cu')
generated.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo',
 '-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v',
 '--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),
 '-I/home/xgao/cutlass/include','-shared',str(generated),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),
 str(root/'csrc/gemm/pearl_gemm_only_sm61.cu'),'-o','/tmp/libpearl-rank.so'],check=True)
(root/'artifacts').mkdir(exist_ok=True)
shutil.copy2('/tmp/libpearl-rank.so', root/'artifacts/libpearl-rank.so')
print('Built isolated library:',root/'artifacts/libpearl-rank.so')
