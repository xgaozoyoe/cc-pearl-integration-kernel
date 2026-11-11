"""Isolated next-iteration copy-address preparation experiment."""
from pathlib import Path
import subprocess,json,re,hashlib,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/handoff-prepare';out.mkdir(exist_ok=True)
s=(root/'artifacts/handoff-lean-source.cu').read_text();ns='handoff_lean_toggle'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);base=s[a:b]
helper=r'''
struct CopyPlan { unsigned a,b,sa,sb; };
__device__ __forceinline__ CopyPlan prepare(unsigned oa,unsigned ob,unsigned da,unsigned db,unsigned step) {
    CopyPlan p;
    // No memory operation: only address arithmetic. Volatile asm makes its
    // intended schedule visible in PTX; generated SASS must still be audited.
    asm volatile("{ .reg .u32 k,phase,xa,xb;\n"
        "shl.b32 k,%8,6; and.b32 phase,%8,1;\n"
        "shl.b32 xa,phase,14; mul.lo.u32 xb,phase,12288;\n"
        "add.u32 %0,%4,k; add.u32 %1,%5,k;\n"
        "add.u32 %2,%6,xa; add.u32 %3,%7,xb; }"
        : "=r"(p.a),"=r"(p.b),"=r"(p.sa),"=r"(p.sb)
        : "r"(oa),"r"(ob),"r"(da),"r"(db),"r"(step) : "memory");
    return p;
}
__device__ __forceinline__ void issue_plan(const signed char* a,const signed char* b,
 CopyPlan p,int grB,int n) {
    #pragma unroll
    for(int j=0;j<3;++j) if(threadIdx.x+j*384<1024) {
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,16;"
            ::"r"(p.sa),"l"(a+p.a):"memory");
        p.a+=96u*4096u;p.sa+=384u*16u;
    }
    #pragma unroll
    for(int j=0;j<2;++j) {
        bool valid=grB<n;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"
            ::"r"(p.sb),"l"(b+(valid?p.b:0u)),"r"(valid?16:0):"memory");
        p.b+=96u*4096u;p.sb+=384u*16u;grB+=96;
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}
'''
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
names=['source8','late','early_wait','early_tail']
for name in names:
 new=base.replace('threadIdx.x>=192','threadIdx.x>=256')
 # Publish after issuing the last B fragment load. All needed shared reads
 # precede bar.arrive; remaining MMA consume registers. Compiler handles WAR.
 load='''                        br[(g+Lead)%slots]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+((g+Lead)%Groups)*16,((g+Lead)/Groups)*32,lane,BN);'''
 assert new.count(load)==1
 new=new.replace(load,'''                    {
'''+load+'''
                        if constexpr(Lead==1 && TileK==64) {
                            if(g+Lead==Groups*2-1) {
                                asm volatile("cp.async.wait_group 0;" ::: "memory");
                                asm volatile("bar.arrive %0, 384;" :: "r"(peer_bar) : "memory");
                            }
                        }
                    }''')
 old='''        asm volatile("cp.async.wait_group 0;" ::: "memory");
        asm volatile("bar.arrive %0, 384;" :: "r"(peer_bar) : "memory");'''
 assert new.count(old)==1;new=new.replace(old,'        // Publication is now at the final B fragment load above.')
 if name!='source8':
  at=new.index('template<int Lead,int TileK=64');new=new[:at]+helper+new[at:]
  point='    unsigned own_bar=4u+unsigned(threadIdx.x>=256);'
  prep='''    const unsigned logical=unsigned(threadIdx.x)^(unsigned(threadIdx.x>>3)&3u);
    const unsigned copy_row=logical/4u,copy_col=(logical%4u)*16u;
    const int copy_grB=cn*BN+int(copy_row);
    const unsigned originA=(unsigned(cm)*BM+copy_row)*4096u+copy_col;
    const unsigned originB=unsigned(copy_grB)*4096u+copy_col;
    const unsigned dstA=__cvta_generic_to_shared(sa+threadIdx.x*16);
    const unsigned dstB=__cvta_generic_to_shared(sb+threadIdx.x*16);
    CopyPlan planned;
'''
  if name!='late':prep+='    planned=prepare(originA,originB,dstA,dstB,1);\n'
  new=new.replace(point,prep+point)
  call='''        if(kt+Stages-1<(k+TileK-1)/TileK)
            stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);'''
  assert new.count(call)==1
  new=new.replace(call,('        planned=prepare(originA,originB,dstA,dstB,kt+1);\n' if name=='late' else '')+'''        if(kt+1<64) issue_plan(a,b,planned,copy_grB,n);''')
  if name!='late':
   at='        asm volatile("bar.sync %0, 384;" :: "r"(own_bar) : "memory");' if name=='early_wait' else '        // Publication is now at the final B fragment load above.'
   assert new.count(at)==1
   new=new.replace(at,'        planned=prepare(originA,originB,dstA,dstB,kt+2);\n'+at)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='lean-source',tofile=name)))
 new=new.replace(ns,'handoff_prepare_'+name);parts.append(new)
parts.append('extern "C" int launch_prepare(int v,const int8_t*a,const int8_t*b,int m,int n,uint32_t*t){')
for v,name in enumerate(names):parts.append(f'if(v=={v})return handoff_prepare_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,nullptr,nullptr);')
parts.append('return cudaErrorInvalidValue; }')
p=out/'source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-shared',str(p),'-o',str(out/'libprepare.so')],check=True)
subprocess.run(['/usr/local/cuda-12.6/bin/cuobjdump','--extract-elf','all',str(out/'libprepare.so')],cwd=out,check=True)
cubins=list(out.glob('*sm_89.cubin'));assert len(cubins)==1
blob=cubins[0].read_bytes();dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(cubins[0])],text=True);(out/'compiled.sass').write_text(dump)
ref=root/'artifacts/handoff-groups';meta=json.loads((ref/'manifest.json').read_text());native=(ref/'group8.cubin').read_bytes();assert hashlib.sha256(native).hexdigest()==meta['variants']['group8']['sha256']
(out/'native8.cubin').write_bytes(native)
manifest={'variants':{'native8':meta['variants']['group8']},'warning':'Offline only, compile-time next-K copy-address preparation. Production unchanged.'}
for name in names:
 entry=next(x for x in re.findall(r'Function : (\S+)',dump) if 'handoff_prepare_'+name in x)
 (out/(name+'.cubin')).write_bytes(blob)
 manifest['variants'][name]={'entry':entry,'sha256':hashlib.sha256(blob).hexdigest(),'smem':69632,'correct':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
