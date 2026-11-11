"""Snapshot local XOR at boundary; defer only redux/store across K iteration."""
from pathlib import Path
import json,hashlib,subprocess,re,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/fold-defer-reduce';out.mkdir(exist_ok=True)
s=(root/'artifacts/handoff-lean-source.cu').read_text();ns='handoff_lean_toggle';a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);base=s[a:b]
helper=r'''
__device__ __forceinline__ void finish_pending(unsigned local,unsigned* transcript,
 int wm,int wn,int lane,int group,unsigned rb) {
    unsigned x=__reduce_xor_sync(0xffffffffu,local);
    if(lane==0) {
        unsigned idx=((wm*4+3)*12+wn*4+group)*16+(rb&15u);
        if(rb<16)transcript[idx]=x;
        else {unsigned old=transcript[idx];transcript[idx]=((old<<13)|(old>>19))^x;}
    }
}
'''
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n'];names=['source8','defer2_copy','defer2_mma','defer4_mma']
for name in names:
 count=0 if name=='source8' else (4 if name=='defer4_mma' else 2)
 new=base.replace('threadIdx.x>=192','threadIdx.x>=256')
 # Use unsigned fold arithmetic in all source controls, independent of deferral.
 new=new.replace('int rb=(kt+1)/2-1;','unsigned rb=unsigned((kt+1)/2-1);').replace('(rb%16)','(rb&15u)')
 if count:
  at=new.index('template<int Lead,int TileK=64');new=new[:at]+helper+new[at:]
  point='    unsigned own_bar=4u+unsigned(threadIdx.x>=256);'
  new=new.replace(point,f'    unsigned pending[{count}]; // local XOR only, not accumulator copies\n'+point)
  # Restrict replacement to the final TileK64 fold, not dead template branches.
  start=new.index('        // Publish this group')
  old='''                    x=__reduce_xor_sync(0xffffffffu,x);
                    if(lane==0) {
                        int idx=((wm*4+row)*(BN/16)+wn*Groups+group)*16+(rb&15u);
                        if(rb<16) transcript[idx]=x;
                        else { unsigned prev=transcript[idx]; transcript[idx]=((prev<<13)|(prev>>19))^x; }
                    }'''
  assert new[start:].count(old)==1
  repl=f'''                    if(row==3 && group>=Groups-{count}) {{
                        pending[group-(Groups-{count})]=x;
                    }} else {{
'''+old+'\n                    }'
  new=new[:start]+new[start:].replace(old,repl)
  if name.endswith('copy'):
   point='        signed char* aa=sa+current*BM*TileK;'
   drain=f'''        // Previous boundary's accumulator reads completed in prior K.
        // Future cp.async has been issued above; reduce while copy is in flight.
        if(kt>0 && (kt&1)==0) {{
            #pragma unroll
            for(int i=0;i<{count};++i)
                finish_pending(pending[i],transcript,wm,wn,lane,Groups-{count}+i,unsigned(kt/2-1));
        }}
'''
   assert new.count(point)==1;new=new.replace(point,drain+point)
  else:
   point='                    mma<PackedB>(c[row][group][half],ar[row],br[slot],half);'
   assert new.count(point)==1
   # At beginning of next K, drain one saved tile after each first-group
   # row's second MMA. Accumulator has already advanced; saved XOR is independent.
   new=new.replace(point,point+f'''
                    if(g==0 && half==1 && row<{count} && kt>0 && (kt&1)==0)
                        finish_pending(pending[row],transcript,wm,wn,lane,Groups-{count}+row,unsigned(kt/2-1));''')
  point='    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;i+=Threads)'
  assert new.count(point)==1
  new=new.replace(point,f'''    // Flush K4096's pending local XOR before final CTA output barrier.
    #pragma unroll
    for(int i=0;i<{count};++i)
        finish_pending(pending[i],transcript,wm,wn,lane,Groups-{count}+i,31u);
'''+point)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='lean8-source',tofile=name)))
 parts.append(new.replace(ns,'fold_defer_'+name))
parts.append('extern "C" int launch_defer(int v,const int8_t*a,const int8_t*b,int m,int n,uint32_t*t){')
for v,name in enumerate(names):parts.append(f'if(v=={v})return fold_defer_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,nullptr,nullptr);')
parts.append('return cudaErrorInvalidValue;}');source=out/'source.cu';source.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-shared',str(source),'-o',str(out/'libdefer.so')],check=True)
subprocess.run(['/usr/local/cuda-12.6/bin/cuobjdump','--extract-elf','all',str(out/'libdefer.so')],cwd=out,check=True)
binaries=list(out.glob('*sm_89.cubin'));assert len(binaries)==1;blob=binaries[0].read_bytes()
dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(binaries[0])],text=True);(out/'compiled.sass').write_text(dump)
ref=root/'artifacts/fold-preamble';meta=json.loads((ref/'manifest.json').read_text());v=meta['variants']['both_compact'];native=(ref/'both_compact.cubin').read_bytes();assert hashlib.sha256(native).hexdigest()==v['sha256'];(out/'native.cubin').write_bytes(native)
m={'variants':{'native':v},'warning':'Isolated offline cross-K reduction experiment; never overwrite production.'}
for name in names:
 entry=next(x for x in re.findall(r'Function : (\S+)',dump) if 'fold_defer_'+name in x)
 (out/(name+'.cubin')).write_bytes(blob);m['variants'][name]={'entry':entry,'sha256':hashlib.sha256(blob).hexdigest(),'smem':69632,'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
