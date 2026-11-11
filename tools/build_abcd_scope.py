"""Complete per-K preparation replay diagnostic; isolated, GPU0-only harness.
Native A/B anchors plus matched B*/C (same kernel, different replay pointer).
This instrument deliberately exposes its trace IO overhead; not a mining backend.
"""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];src=(root/'artifacts/handoff-lean-source.cu').read_text()
def section(ns):
 a=src.index('namespace '+ns);b=src.index('// namespace '+ns,a)+len('// namespace '+ns);return src[a:b]
base=section('handoff_lean_toggle')
helper=r'''
// 41 scalar slots, one prologue step plus64 K iterations. Slot-major lanes
// keep trace IO coalesced. Every computed output is an observable volatile store.
struct Probe {
    unsigned* written; const unsigned* replay;
    size_t lane_index, pitch; int step;
    const signed char* originA; const signed char* originB;
    int firstB;
};
__device__ __forceinline__ unsigned route(Probe p,int slot,unsigned computed) {
    size_t idx=(size_t(p.step)*41+slot)*p.pitch+p.lane_index;
    asm volatile("st.volatile.global.u32 [%0],%1;" :: "l"(p.written+idx),"r"(computed) : "memory");
    if(!p.replay)return computed; // Untimed trace recording only.
    unsigned used;
    asm volatile("ld.volatile.global.u32 %0,[%1];" : "=r"(used) : "l"(p.replay+idx) : "memory");
    return used;
}
__device__ __forceinline__ const signed char* route_ptr(Probe p,int slot,const signed char* ptr) {
    auto bits=reinterpret_cast<unsigned long long>(ptr);
    unsigned lo=route(p,slot,unsigned(bits));
    unsigned hi=route(p,slot+1,unsigned(bits>>32));
    return reinterpret_cast<const signed char*>((static_cast<unsigned long long>(hi)<<32)|lo);
}
'''
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for old,newns in [('handoff_lean_baseline','scope_A'),('handoff_lean_toggle','scope_B')]:parts.append(section(old).replace(old,newns))
for reduced in (False,True):
 s=base;ns='scope_reduced' if reduced else 'scope_probe'
 marker='template<int Layout=0>';s=s.replace(marker,helper+'\n'+marker,1)
 s=s.replace('int lane,int height) {','int lane,int height,Probe probe) {')
 s=s.replace('(base,row,col,lane,height);','(base,row,col,lane,height,probe);')
 marker='    Fragment x;\n    asm volatile("ldmatrix'
 assert s.count(marker)==1
 s=s.replace(marker,'    addr=route(probe,17+(col/32)*4+(row/16)%4,addr);\n'+marker)
 marker='        Fragment x;\n        asm volatile("ldmatrix';assert s.count(marker)==1
 s=s.replace(marker,'        addr=route(probe,25+(col/32)*4+(row/16)%4,addr);\n'+marker)
 s=s.replace('int cm,int cn,int kt,int m,int n,int k) {','int cm,int cn,int kt,int m,int n,int k,Probe probe) {')
 # FixedCopy branch is uninstantiated; remove it so the diagnostic has one path.
 a=s.index('    if constexpr(FixedCopy) {');b=s.index('    static_assert(TileK==64 && Layout==6',a);s=s[:a]+s[b:]
 s=s.replace('if(threadIdx.x+j*THREADS<BM*64/16) {','if(route(probe,38+j,threadIdx.x+j*THREADS<BM*64/16)) {')
 s=s.replace('const signed char* src=a+offA;','const signed char* src=route_ptr(probe,2*j,a+offA);\n        unsigned usedDst=route(probe,6+j,dstA);')
 s=s.replace('"r"(dstA),"l"(src)','"r"(usedDst),"l"(src)')
 s=s.replace('const signed char* src=b+(valid?offB:0u);','const signed char* src=route_ptr(probe,9+2*j,b+(valid?offB:0u));\n        unsigned usedDst=route(probe,13+j,dstB);\n        unsigned usedValid=route(probe,15+j,valid?16:0);')
 s=s.replace('"r"(dstB),"l"(src),"r"(valid?16:0)','"r"(usedDst),"l"(src),"r"(usedValid)')
 s=s.replace('int k,int r,unsigned* out) {','int k,int r,unsigned* out,const unsigned* replay,unsigned* written) {')
 at='    Fragment ar[4], br[Lead+1];';assert s.count(at)==1
 s=s.replace(at,at+'''
    Probe probe{written,replay,(size_t(blockIdx.y)*gridDim.x+blockIdx.x)*THREADS+threadIdx.x,
        size_t(gridDim.x)*gridDim.y*THREADS,0,nullptr,nullptr,0};
''')
 s=s.replace(',cm,cn,p,m,n,k);',',cm,cn,p,m,n,k,probe);')
 s=s.replace('        const unsigned peer_bar=own_bar^1u;','''        probe.step=kt+1;
        unsigned usedOwn=route(probe,35,own_bar);
        unsigned peer_bar=route(probe,36,own_bar^1u);''')
 s=s.replace('int current=kt%Stages,next=(kt+Stages-1)%Stages;','int current=route(probe,33,kt%Stages),next=route(probe,34,(kt+Stages-1)%Stages);')
 s=s.replace('if(kt+Stages-1<(k+TileK-1)/TileK)','if(route(probe,37,kt+Stages-1<(k+TileK-1)/TileK))')
 s=s.replace(',cm,cn,kt+Stages-1,m,n,k);',',cm,cn,kt+Stages-1,m,n,k,probe);')
 s=s.replace(',lane,BM)',',lane,BM,probe)').replace(',lane,BN)',',lane,BN,probe)')
 s=s.replace('"r"(own_bar) : "memory"','"r"(usedOwn) : "memory"')
 s=s.replace('uint32_t* t,int* info,cudaStream_t stream)', 'uint32_t* t,int* info,cudaStream_t stream,const unsigned* replay,unsigned* written)')
 s=s.replace('(a,b,m,n,k,r,t);','(a,b,m,n,k,r,t,replay,written);')
 if reduced:
  # Same output plan; hoist global-copy origins, simplify all A/B swizzles.
  old='return swizzle<Layout>(p);';assert s.count(old)==1
  s=s.replace(old,'''if constexpr(Layout==6 && TileK==64 && !Planar)
        return p ^ ((unsigned(row)&6u)<<3);
    return swizzle<Layout>(p);''')
  start=s.index('    int logical=int(threadIdx.x)');end=s.index('    #pragma unroll\n    for(int j=0;j<3;',start)
  s=s[:start]+'''    unsigned dstA=__cvta_generic_to_shared(sa+threadIdx.x*16);
    unsigned dstB=__cvta_generic_to_shared(sb+threadIdx.x*16);
    const signed char* cursorA=probe.originA+kt*64;
    const signed char* cursorB=probe.originB+kt*64;
    int grB=probe.firstB;
'''+s[end:]
  s=s.replace('route(probe,38+j,threadIdx.x+j*THREADS<BM*64/16)','route(probe,38+j,j<2 || threadIdx.x<256)')
  s=s.replace('route_ptr(probe,2*j,a+offA)','route_ptr(probe,2*j,cursorA)')
  s=s.replace('grA+=THREADS/4;offA+=unsigned(THREADS/4)*unsigned(k);dstA+=THREADS*16;','cursorA+=96*4096;dstA+=THREADS*16;')
  s=s.replace('route_ptr(probe,9+2*j,b+(valid?offB:0u))','route_ptr(probe,9+2*j,valid?cursorB:b)')
  s=s.replace('grB+=THREADS/4;offB+=unsigned(THREADS/4)*unsigned(k);dstB+=THREADS*16;','grB+=96;cursorB+=96*4096;dstB+=THREADS*16;')
  point='    #pragma unroll\n    for(int p=0;p<Stages-1;++p)';assert s.count(point)==1
  s=s.replace(point,'''    unsigned logical=unsigned(threadIdx.x)^(unsigned(threadIdx.x>>3)&3u);
    unsigned row=logical/4, col=(logical%4)*16;
    probe.originA=a+(size_t(cm)*BM+row)*4096+col;
    probe.originB=b+(size_t(cn)*BN+row)*4096+col;
    probe.firstB=cn*BN+row;
'''+point)
 (root/f'artifacts/abcd-scope-{ns}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='validated-lean-toggle',tofile=ns)))
 parts.append(s.replace('handoff_lean_toggle',ns))
parts.append('''extern "C" int scope_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info,const unsigned* trace,unsigned* audit) {
if(m<=0 || n<=0 || m%256 || n%16 || m>8192 || n>8192)return cudaErrorInvalidValue;
''')
for v,ns in [(0,'scope_A'),(1,'scope_B'),(2,'scope_probe'),(3,'scope_probe'),(4,'scope_reduced'),(5,'scope_probe')]:
 extra='' if v<2 else (',nullptr,audit' if v==5 else ',trace,audit' if v==3 else ',audit,audit')
 parts.append(f'if(v=={v})return {ns}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr{extra});')
parts.append('return cudaErrorInvalidValue;\n}')
p=root/'artifacts/abcd-scope-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','-shared',str(p),'-o',str(root/'artifacts/libpearl-abcd-scope.so')],check=True)
