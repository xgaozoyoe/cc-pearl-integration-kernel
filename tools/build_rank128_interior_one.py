"""One-launch CTA-uniform interior specialization; isolated R128 experiment."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/ada-rank-frozen-source.cu').read_text()
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s[s.index('namespace pearl_rank128'):s.index('void launch_pearl_blake3')]
base=s.replace('pearl_rank128','rank128_interior_one_baseline')
new=s.replace('pearl_rank128','rank128_interior_one_fast')
# Compile-time specialization applies to the copy helper only.
new=new.replace('bool FixedCopy=false,int Layout=0>','bool FixedCopy=false,int Layout=0,bool Interior=false>')
new=new.replace('bool valid_row=gr<m &&','bool valid_row=(Interior || gr<m) &&').replace('bool valid_row=gr<n &&','bool valid_row=(Interior || gr<n) &&')
new=new.replace('(gr<m?gr:0)','((Interior || gr<m)?gr:0)').replace('(gr<n?gr:0)','((Interior || gr<n)?gr:0)')
pos=new.index('__global__ void __launch_bounds__')
start=new.rfind('template<',0,pos)
new=new[:start]+new[start:].replace('int Layout=0>','int Layout=0,bool Interior=false>',1)
new=new.replace('__global__ void __launch_bounds__(Wide?256:THREADS,1) kernel','__device__ __forceinline__ void body',1)
new=new.replace('stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>','stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout,Interior>')
pos=new.index('template<int Lead',new.index('    for(int i=threadIdx.x;i<WORDS;'))
wrapper='''
template<int Lead,int TileK=64,bool Planar=false,bool PackedB=false,bool UnrollCopy=false,bool FixedCopy=false,int Stages=2,bool Wide=false,int Layout=0>
__global__ void __launch_bounds__(Wide?256:THREADS,1) kernel(const signed char* a,const signed char* b,int m,int n,int k,int r,unsigned* out) {
    int linear=blockIdx.x+gridDim.x*blockIdx.y;
    int first=linear/(8*gridDim.x)*8;
    int cols=min(8,int(gridDim.y)-first), off=linear%(8*gridDim.x);
    int cm=off/cols,cn=first+off%cols;
    if((cm+1)*BM<=m && (cn+1)*BN<=n)
      body<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout,true>(a,b,m,n,k,r,out);
    else
      body<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout,false>(a,b,m,n,k,r,out);
}
'''
new=new[:pos]+wrapper+new[pos:]
# Reuse tested offline ABI and compile invocation without running other builder.
template=(root/'tools/build_rank128_afull.py').read_text()
start=template.index('source=base+new')
tail=template[start:].replace('afull','interior_one')
tail=tail.replace('    if(v==1 && m%256)return rank128_interior_one_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);\n','')
exec(tail)
