"""Source-instrumented barrier observation; never a mining library."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1];out=root/'artifacts/barrier-timeline';out.mkdir(exist_ok=True)
s=(root/'artifacts/fold-defer-reduce/source.cu').read_text();ns='fold_defer_source8';s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)].replace(ns,'barrier_timeline')
s=s.replace('namespace barrier_timeline {','''namespace barrier_timeline {
__device__ unsigned long long *trace_ptr;
__device__ int trace_block,trace_iteration;
__device__ __forceinline__ unsigned long long stamp() {
 unsigned long long t;asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");return t;
}
''',1)
s=s.replace('template<int Lead,int TileK=64,','template<bool Trace,int Lead,int TileK=64,')
s=s.replace('kernel<Lead,','kernel<Trace,Lead,')
s=s.replace('    unsigned own_bar=4u+unsigned(threadIdx.x>=256);','''    unsigned own_bar=4u+unsigned(threadIdx.x>=256);
    unsigned long long times[5];
''',1)
s=s.replace('        // Publish this group', '''        bool record=false;
        if constexpr(Trace) record=linear==trace_block && kt==trace_iteration;
        if(record) times[0]=stamp();
        // Publish this group''',1)
s=s.replace('        asm volatile("bar.arrive %0, 384;', '        if(record) times[1]=stamp();\n        asm volatile("bar.arrive %0, 384;',1)
s=s.replace('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)', '        if(record) times[2]=stamp();\n        if constexpr(TileK==64) {\n        if((kt+1)%2==0)',1)
s=s.replace('        asm volatile("bar.sync %0, 384;" :: "r"(own_bar) : "memory");','''        if(record) times[3]=stamp();
        asm volatile("bar.sync %0, 384;" :: "r"(own_bar) : "memory");
        if(record) {
            times[4]=stamp();
            if(lane==0) {
                #pragma unroll
                for(int point=0;point<5;++point) trace_ptr[warp*5+point]=times[point];
            }
        }''',1)
# Match the frozen binary: publish before the final16MMA, after all old-stage LDSM.
a=s.index('        bool record=false;');b=s.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)',a)
publish=s[a:b]
publish=publish.replace('        bool record=false;\n        if constexpr(Trace) record=linear==trace_block && kt==trace_iteration;\n','')
s=s[:a]+s[b:]
s=s.replace('        const unsigned peer_bar=own_bar^1u;', '        bool record=false;\n        if constexpr(Trace) record=linear==trace_block && kt==trace_iteration;\n        const unsigned peer_bar=own_bar^1u;',1)
needle='            int slice=g/Groups,group=g%Groups,slot=g%slots;'
assert s.count(needle)==1
s=s.replace(needle,needle+'\n            if(g==Groups*(TileK/32)-1) {\n'+publish+'            }\n',1)
# Make register availability respect the existing native publication boundary.
# The asm does not modify the operands; read/write constraints prevent NVCC
# from pulling the final16MMA across BAR.ARV in this observation build.
args=[f'"+r"(ar[{row}].v[{col}])' for row in range(4) for col in range(4)]+[f'"+r"(br[slot].v[{col}])' for col in range(4)]
s=s.replace('asm volatile("bar.arrive %0, 384;" :: "r"(peer_bar) : "memory");','asm volatile("bar.arrive %20, 384;" : '+','.join(args)+' : "r"(peer_bar) : "memory");')
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s+'''
extern "C" int timeline(int variant,const int8_t*a,const int8_t*b,int m,int n,unsigned*t,unsigned long long*trace,int block,int iteration,int*info) {
 auto err=cudaMemcpyToSymbol(barrier_timeline::trace_ptr,&trace,sizeof(trace));if(err)return err;
 err=cudaMemcpyToSymbol(barrier_timeline::trace_block,&block,sizeof(block));if(err)return err;
 err=cudaMemcpyToSymbol(barrier_timeline::trace_iteration,&iteration,sizeof(iteration));if(err)return err;
 if(variant==0)return barrier_timeline::launch<false,1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 return barrier_timeline::launch<true,1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
}
'''
p=out/'source.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-arch=sm_89','-lineinfo','-Xcompiler','-fPIC','--ptxas-options=-v','-shared',str(p),'-o',str(out/'libtimeline.so')],check=True)
(out/'timeline.sass').write_text(subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/'libtimeline.so')],text=True))
