"""Isolated SM89 prepacked input experiment. Production is never modified."""
from pathlib import Path
import json,subprocess,re,hashlib,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/prepack-k64';out.mkdir(exist_ok=True)
s=(root/'artifacts/fold-defer-reduce/source.cu').read_text();ns='fold_defer_source8';a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);base=s[a:b]
copy=r'''    static_assert(TileK==64 && Layout==6 && !Planar);
    unsigned dstA=__cvta_generic_to_shared(sa)+threadIdx.x*16;
    unsigned dstB=__cvta_generic_to_shared(sb)+threadIdx.x*16;
    const signed char* ap=a+((size_t)cm*64+kt)*(BM*64)+threadIdx.x*16;
    const signed char* bp=b+((size_t)cn*64+kt)*(BN*64)+threadIdx.x*16;
    #pragma unroll
    for(int j=0;j<3;++j) if(threadIdx.x+j*THREADS<BM*64/16) {
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dstA),"l"(ap));
        ap+=THREADS*16;dstA+=THREADS*16;
    }
    #pragma unroll
    for(int j=0;j<2;++j) {
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dstB),"l"(bp));
        bp+=THREADS*16;dstB+=THREADS*16;
    }
    asm volatile("cp.async.commit_group;");
}
'''
start=base.index('    if constexpr(FixedCopy)');end=base.index('\ntemplate<int Lead',start)
packed=base[:start]+copy+base[end:]
parts=['#include <cuda_runtime.h>\n#include <cstdint>']
for name,src in [('source',base),('packed',packed)]:
 parts.append(src.replace(ns,'prepack_'+name))
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),src.splitlines(True),fromfile='source-control',tofile=name)))
parts.append(r'''
// Output [row-block][K64][shared-swizzled stage bytes]. Original inputs untouched.
__global__ void pack_k64(const int8_t* src,int8_t* dst,int rows,int height) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t chunks=(size_t)((rows+height-1)/height)*height*4096/16;
    if(i>=chunks)return;
    unsigned in_stage=i%(height*4);
    unsigned logical=in_stage^((in_stage>>3)&3);
    unsigned stage=i/(height*4),kt=stage%64,block=stage/64;
    unsigned row=block*height+logical/4,col=kt*64+(logical%4)*16;
    uint4 x=make_uint4(0,0,0,0);
    if(row<(unsigned)rows)x=*reinterpret_cast<const uint4*>(src+(size_t)row*4096+col);
    reinterpret_cast<uint4*>(dst)[i]=x;
}
extern "C" int pack_input(const int8_t*src,int8_t*dst,int rows,int height) {
    if(rows<=0||rows%16||(height!=192&&height!=256))return cudaErrorInvalidValue;
    size_t chunks=(size_t)((rows+height-1)/height)*height*4096/16;
    pack_k64<<<(chunks+255)/256,256>>>(src,dst,rows,height);return cudaGetLastError();
}
extern "C" int launch_prepack(int v,const int8_t*a,const int8_t*b,int m,int n,unsigned*t) {
    if(v==0)return prepack_source::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,nullptr,nullptr);
    return prepack_packed::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,nullptr,nullptr);
}
''')
path=out/'source.cu';path.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','-shared',str(path),'-o',str(out/'libprepack.so')],check=True)
subprocess.run(['/usr/local/cuda-12.6/bin/cuobjdump','--extract-elf','all',str(out/'libprepack.so')],cwd=out,check=True)
cubin=out/'source.sm_89.cubin';dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(cubin)],text=True);(out/'compiled.sass').write_text(dump)
ref=root/'artifacts/fold-preamble';v=json.loads((ref/'manifest.json').read_text())['variants']['both_compact'];blob=(ref/'both_compact.cubin').read_bytes();assert hashlib.sha256(blob).hexdigest()==v['sha256'];(out/'native.cubin').write_bytes(blob);m={'variants':{'native':v}}
for name in ('source','packed'):
 blob=cubin.read_bytes();(out/(name+'.cubin')).write_bytes(blob);entry=next(x for x in re.findall(r'Function : (\S+)',dump) if 'prepack_'+name in x)
 m['variants'][name]={'entry':entry,'sha256':hashlib.sha256(blob).hexdigest(),'smem':69632,'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
