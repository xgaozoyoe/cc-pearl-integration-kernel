"""Isolated copy PTX bundle crossed with XOR timing controls; SM89 only."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-xor-control-source.cu').read_text()
def extract(ns):
 a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
 return s[a:b]
copy=r'''
    static_assert(TileK==64 && Layout==6 && !Planar);
    int logical=int(threadIdx.x)^(int(threadIdx.x>>3)&3);
    int row=logical/4,col=(logical%4)*16;
    unsigned da=__cvta_generic_to_shared(sa+threadIdx.x*16);
    unsigned db=__cvta_generic_to_shared(sb+threadIdx.x*16);
    unsigned oa=unsigned(cm*BM+row)*unsigned(k)+unsigned(kt*64+col);
    unsigned ob=unsigned(cn*BN+row)*unsigned(k)+unsigned(kt*64+col);
    // Fixed shared base+immediate offsets avoid incrementing its address
    // between asynchronous copies. Scope entire copy/address block together.
    asm volatile(
        "{ .reg .pred ap,bp; .reg .b32 off,gr,sz; .reg .b64 p0,p1;\n"
        "cvt.u64.u32 p0,%4; add.u64 p0,p0,%0;\n"
        "add.u64 p1,p0,393216;\n"
        "setp.lt.u32 ap,%8,256;\n"
        "cp.async.cg.shared.global [%2],[p0],16;\n"
        "cp.async.cg.shared.global [%2+6144],[p1],16;\n"
        "add.u64 p0,p0,786432;\n"
        "@ap cp.async.cg.shared.global [%2+12288],[p0],16;\n"
        "setp.lt.u32 bp,%6,%7;\n"
        "selp.u32 off,%5,0,bp; selp.u32 sz,16,0,bp;\n"
        "cvt.u64.u32 p0,off; add.u64 p0,p0,%1;\n"
        "cp.async.cg.shared.global [%3],[p0],16,sz;\n"
        "add.u32 gr,%6,96; add.u32 off,%5,393216;\n"
        "setp.lt.u32 bp,gr,%7;\n"
        "selp.u32 off,off,0,bp; selp.u32 sz,16,0,bp;\n"
        "cvt.u64.u32 p1,off; add.u64 p1,p1,%1;\n"
        "cp.async.cg.shared.global [%3+6144],[p1],16,sz;\n"
        "cp.async.commit_group; }\n"
        :: "l"(a),"l"(b),"r"(da),"r"(db),"r"(oa),"r"(ob),
           "r"(cn*BN+row),"r"(n),"r"(unsigned(threadIdx.x)) : "memory");
'''
parts=[extract('xor_control_current').replace('xor_control_current','copy_bundle_original')]
for old,label in [('current','current'),('early','early'),('late','late')]:
 oldns='xor_control_'+old;new=extract(oldns).replace(oldns,'copy_bundle_'+label)
 a=new.index('    static_assert(TileK==64 && Layout==6 && !Planar);');b=new.index('\n}\n',a)
 new=new[:a]+copy+new[b:];parts.append(new)
wrapper='''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int copy_bundle_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(m%256||m>1048576||n>1048576)return cudaErrorInvalidValue;
'''
for v,name in enumerate(['original','current','early','late']):
 wrapper+=f' if(v=={v})return copy_bundle_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);\n'
wrapper+=''' return cudaErrorInvalidValue;
}
extern "C" int copy_bundle_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 auto e=copy_bundle_fold(v,a,b,m,n,t,nullptr);if(e)return e;
 launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''
path=root/'artifacts/rank128-copy-bundle-source.cu';path.write_text('#include <cuda_runtime.h>\n#include <cstdint>\n'+'\n'.join(parts)+wrapper)
(root/'artifacts/rank128-copy-bundle-kernel.diff').write_text(''.join(difflib.unified_diff(parts[0].replace('copy_bundle_original','control').splitlines(True),parts[1].replace('copy_bundle_current','control').splitlines(True),fromfile='original-copy.cu',tofile='bundled-copy.cu')))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),'-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-copy-bundle-experimental.so')],check=True)
