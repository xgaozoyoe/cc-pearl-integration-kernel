"""Two-group fold/MMA overlap on frozen current A-stream; isolated GPU0 ABI."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];ns='rank128_astream_candidate_fast'
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
def block_end(s,a):
 b=s.index('{',a);d=1;e=b+1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return e

def local(row,group):
 return f'''unsigned x=0;
 #pragma unroll
 for(int h=0;h<2;++h)
 #pragma unroll
 for(int v=0;v<4;++v)x^=c[{row}][{group}][h][v];
'''
def publish(value,row,group):
 return f'''{{
 unsigned reduced=__reduce_xor_sync(0xffffffffu,{value});
 int rb=(kt+1)/2-1;
 int idx=((wm*4+({row}))*(BN/16)+wn*Groups+({group}))*16+rb%16;
 unsigned addr=__cvta_generic_to_shared(transcript+idx);
 // Predicate the memory operations without lane-divergent source branches.
 asm volatile("{{{{ .reg .pred owner,old; .reg .b32 prev,value; "
     "setp.eq.u32 owner,%2,0; setp.ge.s32 old,%3,16; and.pred old,old,owner; "
     "mov.b32 value,%1; @old ld.shared.u32 prev,[%0]; "
     "@old shf.l.wrap.b32 prev,prev,prev,13; @old xor.b32 value,prev,%1; "
     "@owner st.shared.u32 [%0],value; }}}}"
     :: "r"(addr),"r"(reduced),"r"(lane),"r"(rb) : "memory");
 }}
'''
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n'];names=['baseline','group8','split8']
for mode,name in enumerate(names):
 new=s
 if mode:
  # Keep the original epilogue only for group B (N16 groups2,3).
  a=new.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)');end=block_end(new,a)
  tail=new[a:end].replace('for(int group=0;group<Groups;++group)','for(int group=2;group<Groups;++group)')
  new=new[:a]+tail+new[end:]
  if mode==2:new=new.replace('    Fragment ar[4], br[Lead+1];','    unsigned pending_x=0;\n    Fragment ar[4], br[Lead+1];')
  point='''                        ar[row]=nextA;
                }
            }
            if constexpr(TileK!=64)'''
  inserted='''                        ar[row]=nextA;
                }
                // In the last K32 slice, groups0/1 are complete before groups2/3.
                if(slice==1 && (kt+1)%2==0 && group>=2) {
                    int tile=(group-2)*4+row;
'''
  if mode==1:
   inserted+=local('tile%4','tile/4')+publish('x','tile%4','tile/4')
  else:
   inserted+='if(tile>0) {\n'+publish('pending_x','(tile-1)%4','(tile-1)/4')+'}\n'
   inserted+=local('tile%4','tile/4')+'pending_x=x;\n'
   inserted+='if(tile==7) {\n'+publish('pending_x','3','1')+'}\n'
  inserted+='''                }
            }
            if constexpr(TileK!=64)'''
  assert new.count(point)==1;new=new.replace(point,inserted)
 (root/f'artifacts/rank128-group-overlap-pred-{name}.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='current A-stream',tofile=name)))
 parts.append(new.replace(ns,'group_overlap_pred_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int group_overlap_pred_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return group_overlap_pred_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int group_overlap_pred_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=group_overlap_pred_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
path=root/'artifacts/rank128-group-overlap-pred-source.cu';path.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-group-overlap-pred-experimental.so')],check=True)
