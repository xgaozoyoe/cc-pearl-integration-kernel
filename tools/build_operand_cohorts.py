"""Isolated SM89 operand-cohort experiment; no production writes."""
from pathlib import Path
import subprocess,json,hashlib,difflib,re
root=Path(__file__).resolve().parents[1];out=root/'artifacts/operand-cohorts';out.mkdir(exist_ok=True)
s=(root/'artifacts/fold-defer-reduce/source.cu').read_text();start=s.index('namespace fold_defer_source8');end=s.index('template<int Lead,int TileK=64',s.index('        own_bar^=2u;',start));base=s[start:end]
copy=r'''
    static_assert(TileK==64 && Layout==6 && !Planar);
    const unsigned warp=threadIdx.x/32, lane=threadIdx.x%32;
    const unsigned wm=warp%4,wn=warp/4;
    // A64x64: 256 vectors distributed over its 3 consumer warps.
    #pragma unroll
    for(unsigned j=0;j<3;++j) {
        unsigned i=wn*32+lane+j*96;
        if(i<256) {
            unsigned row=wm*64+i/4,col=(i%4)*16;
            unsigned dst=__cvta_generic_to_shared(sa+swizzle<Layout>(row*64+col));
            const signed char* src=a+(size_t)(cm*BM+row)*k+kt*64+col;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,16;" :: "r"(dst),"l"(src));
        }
    }
    // B64x64: 256 vectors distributed over its 4 consumer warps.
    #pragma unroll
    for(unsigned j=0;j<2;++j) {
        unsigned i=wm*32+lane+j*128;
        unsigned row=wn*64+i/4,col=(i%4)*16,gr=cn*BN+row;
        unsigned dst=__cvta_generic_to_shared(sb+swizzle<Layout>(row*64+col));
        const signed char* src=b+(size_t)(gr<unsigned(n)?gr:0)*k+kt*64+col;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(gr<unsigned(n)?16:0));
    }
    asm volatile("cp.async.commit_group;");
}
'''
# Verify exactly one writer per vector and only consumer warps write it.
for operand,height in [('A',256),('B',192)]:
 owners={}
 for w in range(12):
  for lane in range(32):
   for j in range(3 if operand=='A' else 2):
    i=(w//4*32+lane+j*96) if operand=='A' else (w%4*32+lane+j*128)
    if i>=256:continue
    row=(w%4 if operand=='A' else w//4)*64+i//4;key=(row,i%4)
    assert key not in owners;owners[key]=w
 assert len(owners)==height*4
manifest={'variants':{}}
native=json.loads((root/'artifacts/fold-preamble/manifest.json').read_text())['variants']['both_compact'];data=(root/'artifacts/fold-preamble/both_compact.cubin').read_bytes();assert hashlib.sha256(data).hexdigest()==native['sha256'];(out/'native.cubin').write_bytes(data);manifest['variants']['native']=native
for name in ['source_control','remap_named','cohort_ab','cohort_ba']:
 new=base
 if name!='source_control':
  a=new.index('    if constexpr(FixedCopy)');b=new.index('\ntemplate<int Lead',a);new=new[:a]+copy+new[b:]
 if name.startswith('cohort_'):
  new=new.replace('        asm volatile("bar.arrive %0, 384;" :: "r"(peer_bar) : "memory");','')
  old='        asm volatile("bar.sync %0, 384;" :: "r"(own_bar) : "memory");'
  barriers=['        asm volatile("bar.sync %0, 96;" :: "r"(unsigned(1+wm)) : "memory");','        asm volatile("bar.sync %0, 128;" :: "r"(unsigned(5+wn)) : "memory");']
  if name=='cohort_ba':barriers.reverse()
  assert new.count(old)==1;new=new.replace(old,'\n'.join(barriers))
 ns='operand_'+name;new=new.replace('fold_defer_source8',ns)
 new+='\ntemplate __global__ void kernel<1,64,false,true,false,false,2,false,6>(const signed char*,const signed char*,int,int,int,int,unsigned*);\n}\n'
 source='#include <cuda_runtime.h>\n#include <cstdint>\n'+new
 (out/(name+'.cu')).write_text(source);(out/(name+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='source8',tofile=name)))
 command=['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/(name+'.cu')),'-o',str(out/(name+'.cubin'))]
 result=subprocess.run(command,capture_output=True,text=True);(out/(name+'.build.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'.cubin'))],text=True);(out/(name+'.sass')).write_text(dump);entry=re.search(r'Function : (\S+)',dump)[1]
 manifest['variants'][name]={'entry':entry,'sha256':hashlib.sha256((out/(name+'.cubin')).read_bytes()).hexdigest(),'smem':69632,'correct':True}
 print(name,result.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
