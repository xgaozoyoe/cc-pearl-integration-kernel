"""Native D: same complete preparation outputs, less address arithmetic, no replay IO."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];s=(root/'artifacts/abcd-scope-source.cu').read_text()
def section(ns):
 a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);return s[a:b]
b=section('scope_B');d=section('scope_reduced')
a=d.index('    size_t idx=',d.index('unsigned route('));z=d.index('\n}',a);d=d[:a]+'    return computed; // No trace, audit, or replay operations in native D.\n'+d[z:]
d=d.replace('const signed char* originA; const signed char* originB;','unsigned originA; unsigned originB;').replace('0,nullptr,nullptr,0};','0,0,0,0};')
d=d.replace('const signed char* cursorA=probe.originA+kt*64;','unsigned cursorA=probe.originA+kt*64;').replace('const signed char* cursorB=probe.originB+kt*64;','unsigned cursorB=probe.originB+kt*64;')
d=d.replace('route_ptr(probe,2*j,cursorA)','route_ptr(probe,2*j,a+cursorA)').replace('valid?cursorB:b','valid?b+cursorB:b')
d=d.replace('probe.originA=a+(size_t(cm)*BM+row)*4096+col;','probe.originA=(unsigned(cm)*BM+row)*4096+col;').replace('probe.originB=b+(size_t(cn)*BN+row)*4096+col;','probe.originB=(unsigned(cn)*BN+row)*4096+col;')
(root/'artifacts/abcd-native-d.diff').write_text(''.join(difflib.unified_diff(b.splitlines(True),d.splitlines(True),fromfile='native-B',tofile='native-D')))
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+b+'\n'+d+'\n'+'''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int scope_d_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>8192 || n>8192)return cudaErrorInvalidValue;
if(v==0)return scope_B::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
if(v==1)return scope_reduced::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr,nullptr,nullptr);
return cudaErrorInvalidValue;
}
extern "C" int scope_d_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=scope_d_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''
p=root/'artifacts/abcd-native-d-source.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','-Icsrc/gemm','-Icsrc','-I/home/xgao/cutlass/include','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-abcd-native-d.so')],check=True)
