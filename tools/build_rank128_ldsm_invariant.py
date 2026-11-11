"""Hoist K64 LDSM swizzle component, preserve operand mapping and load count."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];ns='rank128_astream_candidate_fast'
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
# Verify row16 strides leave low swizzle bits intact, and K32 slice is XOR32.
checks=0
for warp in range(12):
 for lane in range(32):
  for block in range(4):
   for half in range(2):
    for operand in ('A','B'):
     row=(warp%4)*64+(lane&15) if operand=='A' else (warp//4)*64+(lane&7)+(lane>>4)*8
     col=(lane>>4)*16 if operand=='A' else ((lane>>3)&1)*16
     def swz(p):return p^((p>>3)&0x30)
     assert swz((row+block*16)*64+col+half*32)==(swz(row*64+col)^(half*32))+block*1024
     checks+=1
print('Address identities checked:',checks,flush=True)
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n'];names=['baseline','a','ab']
for mode,name in enumerate(names):
 new=s
 if mode:
  helper='''
__device__ __forceinline__ Fragment load_address(unsigned addr) {
 Fragment x;
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
 : "=r"(x.v[0]),"=r"(x.v[1]),"=r"(x.v[2]),"=r"(x.v[3]) : "r"(addr));
 return x;
}
'''
  i=new.index('template<bool Packed=false>');new=new[:i]+helper+new[i:]
  invariant='''    static_assert(TileK==64 && !Planar && PackedB && Layout==6);
    unsigned a_lane_offset=rank128_astream_candidate_fast::offset<64,false,6>(wm*64+(lane&15),(lane>>4)*16,BM);
'''
  if mode==2:invariant+='    unsigned b_lane_offset=rank128_astream_candidate_fast::offset<64,false,6>(wn*64+(lane&7)+(lane>>4)*8,((lane>>3)&1)*16,BN);\n'
  new=new.replace('    unsigned c[4][Groups][2][4]={};',invariant+'    unsigned c[4][Groups][2][4]={};')
  replacements={
   'load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,0,lane,BM)':'load_address(__cvta_generic_to_shared(aa)+a_lane_offset+row*1024)',
   'load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,(slice+1)*32,lane,BM)':'load_address(__cvta_generic_to_shared(aa)+(a_lane_offset^((slice+1)*32))+row*1024)'}
  if mode==2:
   replacements.update({
    'loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+p*16,0,lane,BN)':'load_address(__cvta_generic_to_shared(bb)+b_lane_offset+p*1024)',
    'loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+((g+Lead)%Groups)*16,((g+Lead)/Groups)*32,lane,BN)':'load_address(__cvta_generic_to_shared(bb)+(b_lane_offset^(((g+Lead)/Groups)*32))+((g+Lead)%Groups)*1024)'})
  for old,value in replacements.items():assert new.count(old)==1,old;new=new.replace(old,value)
 (root/f'artifacts/rank128-ldsm-invariant-{name}.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='A-stream baseline',tofile=name)))
 parts.append(new.replace(ns,'ldsm_invariant_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int ldsm_invariant_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return ldsm_invariant_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int ldsm_invariant_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=ldsm_invariant_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
path=root/'artifacts/rank128-ldsm-invariant-source.cu';path.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-ldsm-invariant-experimental.so')],check=True)
