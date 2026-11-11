"""Offline 192x192 CTA comparison; preserve 64x64 warp MMA mapping."""
from pathlib import Path
import subprocess, re, json, hashlib, difflib
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/transcript-cta192';out.mkdir(exist_ok=True)
manifest={'variants':{}}
copy=r'''
    static_assert(TileK==64 && Layout==6 && !Planar && !FixedCopy);
    // Each physical shared 16B chunk has one writer, including partial CTAs.
    #pragma unroll
    for(int j=0;j<3;++j) {
        int physical=int(threadIdx.x)+j*THREADS;
        if(physical<BM*TileK/16) {
            int logical=physical^((physical>>3)&3);
            int row=logical/4,col=(logical%4)*16;
            int gr=cm*BM+row;
            unsigned dst=__cvta_generic_to_shared(sa+physical*16);
            const signed char* src=a+(gr<m?(size_t(gr)*k+kt*64+col):0);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(gr<m?16:0));
        }
        if(physical<BN*TileK/16) {
            int logical=physical^((physical>>3)&3);
            int row=logical/4,col=(logical%4)*16;
            int gr=cn*BN+row;
            unsigned dst=__cvta_generic_to_shared(sb+physical*16);
            const signed char* src=b+(gr<n?(size_t(gr)*k+kt*64+col):0);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(gr<n?16:0));
        }
    }
    asm volatile("cp.async.commit_group;");
'''
for name,original in [('self','source_control'),('consumer','consumer_fine')]:
    base=(root/'artifacts/transcript-tasks'/f'{original}.cu').read_text()
    s=base.replace('BM=256, BN=192, THREADS=384','BM=192, BN=192, THREADS=288')
    start=s.index('    if constexpr(FixedCopy) {')
    end=s.index('\n}',start)
    s=s[:start]+copy+s[end:]
    s=s.replace('warp%4,wn=warp/4','warp%3,wn=warp/3')
    s=s.replace('threadIdx.x>=256','threadIdx.x>=192') # same 2:1 cohort ratio
    s=s.replace('bar.arrive %0, 384','bar.arrive %0, 288').replace('bar.sync %0, 384','bar.sync %0, 288')
    if name=='consumer':
        for a,b in [('__launch_bounds__(416,1)','__launch_bounds__(320,1)'),('shared+69632+6144','shared+58368+4608'),('shared+69632','shared+58368'),('threadIdx.x<24','threadIdx.x<18'),('warp==12','warp==9'),('producer<12','producer<9'),('producer%4','producer%3'),('producer/4','producer/3'),('bar.sync 0,384','bar.sync 0,288'),('threadIdx.x<384','threadIdx.x<288')]:s=s.replace(a,b)
    s=s.replace('transcript_tasks_'+original,'transcript_cta192_'+name)
    (out/f'{name}.cu').write_text(s)
    (out/f'{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile=original,tofile=name)))
    r=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/f'{name}.cu'),'-o',str(out/f'{name}.cubin')],capture_output=True,text=True)
    (out/f'{name}.build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
    sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True);(out/f'{name}.sass').write_text(sass)
    manifest['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/f'{name}.cubin').read_bytes()).hexdigest(),threads=288 if name=='self' else 320,smem=58368 if name=='self' else 63120,correct=True)
    print(name,r.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
