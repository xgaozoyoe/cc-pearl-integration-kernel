"""Stage local XORs early; retain REDUX/transcript epilogue. Isolated SM89 trials."""
from pathlib import Path
import argparse,subprocess,difflib
p=argparse.ArgumentParser();p.add_argument('groups',type=int,choices=[1,2,4]);args=p.parse_args()
root=Path(__file__).resolve().parents[1];name=f'fold_local_late{args.groups}'
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();ns='rank128_astream_candidate_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);s=s[a:b]
base=s.replace(ns,'rank128_'+name+'_baseline');new=s.replace(ns,'rank128_'+name+'_fast')
helper=r'''
__device__ __forceinline__ unsigned local_xor(const unsigned* a,const unsigned* b) {
 unsigned x;
 // Volatile PTX prevents moving the local fold back into the REDUX epilogue.
 asm volatile("{ .reg .b32 p,q;\n"
              "lop3.b32 p,%1,%2,%3,0x96;\n"
              "lop3.b32 q,%4,%5,%6,0x96;\n"
              "lop3.b32 p,p,%7,%8,0x96;\n"
              "xor.b32 %0,p,q; }"
              : "=r"(x) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
                           "r"(b[0]),"r"(b[1]),"r"(b[2]),"r"(b[3]));
 return x;
}
'''
point='template<int Lead,int TileK=64,bool Planar=false,bool PackedB=false,bool UnrollCopy=false,bool FixedCopy=false,int Stages=2,bool Wide=false,int Layout=0>\n__global__'
assert new.count(point)==1;new=new.replace(point,helper+'\n'+point)
point='    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {'
assert new.count(point)==1;new=new.replace(point,point+f'\n        unsigned partial[4][{args.groups}];')
point='''                        ar[row]=nextA;
                }
            }
            if constexpr(TileK!=64)'''
replacement='''                        ar[row]=nextA;
                }
                // One fragment of independent MMA separates its producer
                // from local XOR. No REDUX or transcript access here.
                if(slice==1 && (kt+1)%2==0 && group*4+row>0) {
                    int pr=(row+3)%4,pg=group-(row==0);
                    if(pg>=Groups-EARLY)partial[pr][pg-(Groups-EARLY)]=local_xor(c[pr][pg][0],c[pr][pg][1]);
                }
                if(slice==1 && (kt+1)%2==0 && group==Groups-1 && row==3) {
                    if(group>=Groups-EARLY)partial[row][group-(Groups-EARLY)]=local_xor(c[row][group][0],c[row][group][1]);
                }
            }
            if constexpr(TileK!=64)'''.replace('EARLY',str(args.groups))
assert new.count(point)==1;new=new.replace(point,replacement)
a=new.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)');tail=new[a:]
point='''                    unsigned x=0;
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int v=0;v<4;++v) x^=c[row][group][half][v];
                    }'''
assert tail.count(point)==1
tail=tail.replace(point,f'''                    unsigned x;
                    if(group>=Groups-{args.groups})x=partial[row][group-(Groups-{args.groups})];
                    else {{
                        x=0;
                        #pragma unroll
                        for(int half=0;half<2;++half) {{
                            #pragma unroll
                            for(int v=0;v<4;++v)x^=c[row][group][half][v];
                        }}
                    }}''')
new=new[:a]+tail
wrapper=r'''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int NAME_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(m%256||m>1048576||n>1048576)return cudaErrorInvalidValue;
 if(v==0)return rank128_NAME_baseline::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 if(v==1)return rank128_NAME_fast::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 return cudaErrorInvalidValue;
}
extern "C" int NAME_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 auto e=NAME_fold(v,a,b,m,n,t,nullptr);if(e)return e;
 launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''.replace('NAME',name)
path=root/f'artifacts/rank128-{name}-source.cu';path.write_text('#include <cuda_runtime.h>\n#include <cstdint>\n'+base+'\n'+new+wrapper)
a=base.replace('rank128_'+name+'_baseline','rank128_trial');b=new.replace('rank128_'+name+'_fast','rank128_trial')
(root/f'artifacts/rank128-{name}-kernel.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='current-astream.cu',tofile='early-local-xor.cu')))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),'-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/f'artifacts/libpearl-rank128-{name}-experimental.so')],check=True)
