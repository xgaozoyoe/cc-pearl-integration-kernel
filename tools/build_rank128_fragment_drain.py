"""Small cross-K-iteration fragment pipeline, separate from mining libraries."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();ns='rank128_astream_candidate_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);s=s[a:b]
base=s.replace(ns,'fragment_drain_original');parts=[base]
for count in [1,2]:
 new=s.replace(ns,f'fragment_drain_{count}')
 point='        signed char* bb=sb+current*BN*TileK;'
 capture=f'''
        // Snapshot only deferred previous-boundary fragments before ANY new
        // MMA can overwrite them. Pending scalars survive operand loads/MMA.
        unsigned pending[{count}];
        if(kt>0 && kt%2==0) {{
            #pragma unroll
            for(int pr=0;pr<{count};++pr) {{
                unsigned x;
                asm volatile("{{ .reg .b32 p,q; "
                    "lop3.b32 p,%1,%2,%3,0x96; "
                    "lop3.b32 q,%4,%5,%6,0x96; "
                    "lop3.b32 p,p,%7,%8,0x96; "
                    "xor.b32 %0,p,q; }}"
                    : "=r"(x) : "r"(c[pr][0][0][0]),"r"(c[pr][0][0][1]),
                      "r"(c[pr][0][0][2]),"r"(c[pr][0][0][3]),
                      "r"(c[pr][0][1][0]),"r"(c[pr][0][1][1]),
                      "r"(c[pr][0][1][2]),"r"(c[pr][0][1][3]));
                pending[pr]=x;
            }}
        }}
'''
 assert new.count(point)==1;new=new.replace(point,point+capture)
 point='''                        ar[row]=nextA;
                }
            }
            if constexpr(TileK!=64)'''
 replacement=f'''                        ar[row]=nextA;
                }}
                // New-iteration MMA has advanced while the old boundary
                // survives in pending; publish ONLY that saved old value.
                if(slice==0 && group==0 && row>={count} && row<{2*count} && kt>0 && kt%2==0) {{
                    unsigned x=__reduce_xor_sync(0xffffffffu,pending[row-{count}]);
                    if(lane==0) {{
                        int rb=kt/2-1,pr=row-{count};
                        int idx=((wm*4+pr)*(BN/16)+wn*Groups)*16+rb%16;
                        if(rb<16)transcript[idx]=x;
                        else {{unsigned old=transcript[idx];transcript[idx]=((old<<13)|(old>>19))^x;}}
                    }}
                }}
            }}
            if constexpr(TileK!=64)'''
 assert new.count(point)==1;new=new.replace(point,replacement)
 a=new.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)');tail=new[a:]
 point='''                for(int group=0;group<Groups;++group) {
                    unsigned x=0;'''
 assert tail.count(point)==1
 tail=tail.replace(point,f'''                for(int group=0;group<Groups;++group) {{
                    // No next iteration after K4096: drain everything there.
                    if(group==0 && row<{count})continue;
                    unsigned x=0;''')
 new=new[:a]+tail
 drain=f'''
    // Drain deferred K4096 boundary once, outside the steady-state loop.
    #pragma unroll
    for(int pr=0;pr<{count};++pr) {{
        unsigned x=0;
        #pragma unroll
        for(int h=0;h<2;++h) {{
            #pragma unroll
            for(int v=0;v<4;++v)x^=c[pr][0][h][v];
        }}
        x=__reduce_xor_sync(0xffffffffu,x);
        if(lane==0) {{
            int idx=((wm*4+pr)*(BN/16)+wn*Groups)*16+15;
            unsigned old=transcript[idx];transcript[idx]=((old<<13)|(old>>19))^x;
        }}
    }}
'''
 point='    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;'
 assert new.count(point)==1;new=new.replace(point,drain+point)
 parts.append(new)
 (root/f'artifacts/rank128-fragment-drain-{count}.diff').write_text(''.join(difflib.unified_diff(base.replace('fragment_drain_original','trial').splitlines(True),new.replace(f'fragment_drain_{count}','trial').splitlines(True),fromfile='current-astream.cu',tofile=f'defer-{count}-fragments.cu')))
wrapper='''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int fragment_drain_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(m%256||m>1048576||n>1048576)return cudaErrorInvalidValue;
'''
for v,label in enumerate(['original','1','2']):wrapper+=f' if(v=={v})return fragment_drain_{label}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);\n'
wrapper+=''' return cudaErrorInvalidValue;
}
extern "C" int fragment_drain_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 auto e=fragment_drain_fold(v,a,b,m,n,t,nullptr);if(e)return e;
 launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''
path=root/'artifacts/rank128-fragment-drain-source.cu';path.write_text('#include <cuda_runtime.h>\n#include <cstdint>\n'+'\n'.join(parts)+wrapper)
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),'-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-fragment-drain-experimental.so')],check=True)
