"""Offline address-cost controls for the existing named-region protocol."""
from pathlib import Path
import subprocess,json,hashlib,re,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/named-a0-address';out.mkdir(exist_ok=True)
parent=root/'artifacts/named-a0-publish';meta=json.loads((parent/'manifest.json').read_text())
manifest={'variants':{}}
for n in ('native','publish_after_a'):
    v=meta['variants'][n];b=(parent/f'{n}.cubin').read_bytes();assert hashlib.sha256(b).hexdigest()==v['sha256']
    (out/f'{n}.cubin').write_bytes(b);manifest['variants'][n]=v
base=(parent/'publish_after_a.cu').read_text()
for name in ('offset32','shared_address'):
    s=base
    if name=='offset32':
        old='    const signed char* src=a+(size_t(cm*BM)+row)*k+copy_k*64+col;'
        assert old in s
        s=s.replace(old,'    unsigned off=(unsigned(cm*BM)+row)*unsigned(k)+unsigned(copy_k*64)+col;\n    const signed char* src=a+off;')
    else:
        # Compute A/B copy addresses once. Move early copy and old handoff
        # into this same routine; preserve all copy counts and region signals.
        s=s.replace('int cm,int cn,int kt,int m,int n,int k) {','int cm,int cn,int kt,int m,int n,int k,unsigned handoff=0) {')
        needle='    unsigned offB=unsigned(grB)*unsigned(k)+unsigned(kt*64+col);\n'
        assert s.count(needle)==1
        s=s.replace(needle,needle+'''    if(kt>=2) {
        if(threadIdx.x>=128 && threadIdx.x<256) {
            unsigned free_id=1u+unsigned((kt-2)&1);
            asm volatile("barrier.cta.sync %0,192;" :: "r"(free_id) : "memory");
            const signed char* src=a+offA;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,16;"
                         :: "r"(dstA),"l"(src) : "memory");
        }
        asm volatile("bar.sync %0,384;" :: "r"(handoff) : "memory");
    }
''')
        a=s.index('// Offline experiment only;');b=s.index('template<int Lead,int TileK=64',a)
        s=s[:a]+s[b:]
        a=s.index('        if(kt>0) {',s.index('    for(int kt=0;'))
        b=s.index('        signed char* aa=',a)
        s=s[:a]+'''        int current=kt%Stages,next=(kt+Stages-1)%Stages;
        if(kt+1<k/64)
            stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+1,m,n,k,own_bar);
        else if(kt>0)
            asm volatile("bar.sync %0,384;" :: "r"(own_bar) : "memory");
        if(kt>0) own_bar^=2u;
        const unsigned peer_bar=own_bar^1u;
'''+s[b:]
    s=s.replace('named_a0_publish_after_a','named_address_'+name)
    path=out/f'{name}.cu';path.write_text(s)
    (out/f'{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='publish_after_a',tofile=name)))
    p=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(path),'-o',str(out/f'{name}.cubin')],capture_output=True,text=True)
    (out/f'{name}.build.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
    sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True)
    (out/f'{name}.sass').write_text(sass)
    manifest['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/f'{name}.cubin').read_bytes()).hexdigest(),smem=69632,correct=True)
    print(name,p.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
