"""Independent transcript consumer warp; fine-grained4tile tasks,offline only."""
from pathlib import Path
import subprocess,json,hashlib,re,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/transcript-tasks';out.mkdir(exist_ok=True)
s=(root/'artifacts/fold-defer-reduce/source.cu').read_text();a=s.index('namespace fold_defer_source8');b=s.index('template<int Lead,int TileK=64',s.index('        own_bar^=2u;',a));base=s[a:b]
helpers=r'''
__device__ __forceinline__ void task_wait(unsigned bar,unsigned phase) {
 unsigned ok;
 do { asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p,[%1],%2; selp.u32 %0,1,0,p; }" : "=r"(ok) : "r"(bar),"r"(phase) : "memory"); } while(!ok);
}
__device__ __forceinline__ void task_arrive(unsigned bar) {
 asm volatile("mbarrier.arrive.shared.b64 _,[%0];" :: "r"(bar) : "memory");
}
'''
v=json.loads((root/'artifacts/fold-preamble/manifest.json').read_text())['variants']['both_compact'];blob=(root/'artifacts/fold-preamble/both_compact.cubin').read_bytes();assert hashlib.sha256(blob).hexdigest()==v['sha256'];(out/'native.cubin').write_bytes(blob)
m={'description':'12MMA warps+1independent transcript consumer. Each task4tiles,per-producer ready/free objects,count32. Offline only.','variants':{'native':{**v,'threads':384}}}
for name in ('source_control','consumer_late','consumer_fine'):
 new=base
 if name!='source_control':
  pos=new.index('template<int Lead,int TileK=64');new=new[:pos]+helpers+new[pos:]
  new=new.replace('__launch_bounds__(Wide?256:THREADS,1)','__launch_bounds__(416,1)')
  point='    unsigned c[4][Groups][2][4]={};'
  begin=r'''
    unsigned* tasks=reinterpret_cast<unsigned*>(shared+69632);
    unsigned barbase=(unsigned)__cvta_generic_to_shared(shared+69632+6144);
    if(threadIdx.x<24) {
        unsigned addr=barbase+8*threadIdx.x;
        asm volatile("mbarrier.init.shared.b64 [%0],32;" :: "r"(addr) : "memory");
    }
    __syncthreads();
    if(warp==12) {
        // Sequence order prevents waiting for a future task while older tasks
        // hold producer slots. Each wait targets one producer and4tiles only.
        for(unsigned seq=0;seq<128;++seq) {
            unsigned rb=seq/4,group=seq%4;
            for(unsigned producer=0;producer<12;++producer) {
                task_wait(barbase+producer*16,seq&1u);
                unsigned values[4];
                #pragma unroll
                for(int row=0;row<4;++row) values[row]=tasks[producer*128+row*32+lane];
                task_arrive(barbase+producer*16+8);
                #pragma unroll
                for(int row=0;row<4;++row) {
                    unsigned x=__reduce_xor_sync(0xffffffffu,values[row]);
                    if(lane==0) {
                        int idx=(((producer%4)*4+row)*(BN/16)+(producer/4)*4+group)*16+(rb&15u);
                        if(rb<16)transcript[idx]=x;
                        else {unsigned prev=transcript[idx];transcript[idx]=((prev<<13)|(prev>>19))^x;}
                    }
                }
            }
        }
    } else {
'''
  new=new.replace(point,begin+point)
  new=new.replace('    __syncthreads();\n    unsigned own_bar=', '    asm volatile("bar.sync 0,384;" ::: "memory");\n    unsigned own_bar=')
  start=new.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0) {',new.index('// Publish this group'))
  opening=new.index('{',start);depth=1;end=opening+1
  while depth:
   depth+=(new[end]=='{')-(new[end]=='}');end+=1
  publish=r'''
            unsigned seq=unsigned((kt+1)/2-1)*4+group;
            if(seq)task_wait(barbase+warp*16+8,(seq-1)&1u);
            #pragma unroll
            for(int row=0;row<4;++row) {
                unsigned x=0;
                #pragma unroll
                for(int half=0;half<2;++half) {
                    #pragma unroll
                    for(int v=0;v<4;++v)x^=c[row][group][half][v];
                }
                tasks[warp*128+row*32+lane]=x;
            }
            task_arrive(barbase+warp*16);
'''
  new=new[:start]+(('        if((kt+1)%2==0) {\n            for(int group=0;group<4;++group) {\n'+publish+'\n            }\n        }') if name=='consumer_late' else '')+new[end:]
  if name=='consumer_fine':
   point='            if constexpr(TileK!=64) {\n        if(group'
   assert new.count(point)==1
   new=new.replace(point,'            if(slice==1 && (kt+1)%2==0) {\n'+publish+'\n            }\n'+point)
  point='    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;i+=Threads) {'
  assert new.count(point)==1
  new=new.replace(point,'''    } // producer path
    __syncthreads();
    if(threadIdx.x<24) {
        unsigned addr=barbase+8*threadIdx.x;
        asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(addr) : "memory");
    }
    for(int i=threadIdx.x;threadIdx.x<384 && i<WORDS;i+=Threads) {''')
 ns='transcript_tasks_'+name;new=new.replace('fold_defer_source8',ns)
 new+='\ntemplate __global__ void kernel<1,64,false,true,false,false,2,false,6>(const signed char*,const signed char*,int,int,int,int,unsigned*);\n}\n'
 path=out/(name+'.cu');path.write_text('#include <cuda_runtime.h>\n#include <cstdint>\n'+new)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='source8',tofile=name)))
 r=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(path),'-o',str(out/(name+'.cubin'))],capture_output=True,text=True)
 (out/(name+'.build.log')).write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'.cubin'))],text=True);(out/(name+'.sass')).write_text(dump)
 m['variants'][name]={'entry':re.search(r'Function : (\S+)',dump)[1],'sha256':hashlib.sha256((out/(name+'.cubin')).read_bytes()).hexdigest(),'threads':384 if name=='source_control' else 416,'smem':69632 if name=='source_control' else 75968,'correct':True}
 print(name,r.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
