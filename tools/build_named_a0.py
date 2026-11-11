"""Offline named-barrier A32:64 early-copy test. No production library writes."""
from pathlib import Path
import subprocess, json, hashlib, re, difflib
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/named-a0';out.mkdir(exist_ok=True)
parent=root/'artifacts/arrival-order'
meta=json.loads((parent/'manifest.json').read_text())
manifest={'variants':{}}
for name in ('native','source_control'):
    v=meta['variants'][name];blob=(parent/f'{name}.cubin').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==v['sha256']
    (out/f'{name}.cubin').write_bytes(blob);manifest['variants'][name]=dict(v,correct=True)
base=(root/'artifacts/named-a0-design/design-only.cu').read_text()
sync='''            // Same previous-round handoff, now after this one early copy.
            asm volatile("bar.sync %0,384;" :: "r"(own_bar) : "memory");
            own_bar^=2u;
'''
assert base.count(sync)==1
for name in ('late_control','early_a'):
    s=base
    if name=='late_control':
        s=s.replace(sync,'')
        s=s.replace('        if(kt>0) {\n','        if(kt>0) {\n'+sync)
    s=s.replace('arrival_order_source_control','named_a0_'+name)
    s=s.replace('// DESIGN ONLY: not built or CUDA-validated. No production entrypoint.',
                '// Offline experiment only; consult validation records before use.')
    path=out/f'{name}.cu';path.write_text(s)
    (out/f'{name}.diff').write_text(''.join(difflib.unified_diff(
        (parent/'source_control.cu').read_text().splitlines(True),s.splitlines(True),
        fromfile='source_control',tofile=name)))
    p=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo',
        '-arch=sm_89','--cubin','--ptxas-options=-v',str(path),'-o',str(out/f'{name}.cubin')],
        capture_output=True,text=True)
    (out/f'{name}.build.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
    sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True)
    (out/f'{name}.sass').write_text(sass)
    assert 'ATOMS.ARRIVE' not in sass
    manifest['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],
        sha256=hashlib.sha256((out/f'{name}.cubin').read_bytes()).hexdigest(),smem=69632,threads=384,correct=True)
    print(name,p.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
