"""Build opt-in R128 A-stream trial without replacing any existing mining library."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'tools/build_rank128_astream.py'),'1'],check=True)
s=Path('/tmp/rank128-astream1-generated.cu').read_text()
# Keep CUDA registrations and template initialization unique across loaded DSOs.
s=s.replace('astream1', 'astream_candidate')
s+='''
extern "C" int rank_fold(int r,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(r!=128 || m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
 return astream_candidate_fold(1,a,b,m,n,t,info);
}
extern "C" int rank_pow(int r,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 if(r!=128 || m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
 return astream_candidate_pow(1,a,b,m,n,key,target,t,dig,found,coord);
}
'''
path=root/'artifacts/rank128-astream-candidate-source.cu';path.write_text(s)
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo',
 '-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v',
 '--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),
 '-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),
 '-o',str(root/'artifacts/libpearl-rank128-astream-candidate.so')],check=True)
