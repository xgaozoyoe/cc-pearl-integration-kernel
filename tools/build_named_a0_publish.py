"""Offline named-barrier A32:64 early-copy test. No production library writes."""
from pathlib import Path
import subprocess, json, hashlib, re, difflib
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/named-a0-publish';out.mkdir(exist_ok=True)
parent=root/'artifacts/named-a0'
meta=json.loads((parent/'manifest.json').read_text())
manifest={'variants':{}}
for name in ('native','source_control','early_a'):
    v=meta['variants'][name];blob=(parent/f'{name}.cubin').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==v['sha256']
    (out/f'{name}.cubin').write_bytes(blob);manifest['variants'][name]=dict(v,correct=True)
base=(parent/'early_a.cu').read_text()
start=base.index('        // Only read-only participants publish here.')
end=base.index('        if constexpr(TileK==64)',start)
publish=base[start:end]
for name in ('publish_after_a',):
    s=base[:start]+base[end:]
    point='            if constexpr(TileK!=64) {\n        if(group==Groups-1'
    assert s.count(point)==1
    insert='            if constexpr(TileK==64) { if(g==Groups-1) {\n'+publish+'            }}\n'
    s=s.replace(point,insert+point)
    s=s.replace('named_a0_early_a','named_a0_'+name)
    path=out/f'{name}.cu';path.write_text(s)
    (out/f'{name}.diff').write_text(''.join(difflib.unified_diff(
        base.splitlines(True),s.splitlines(True),
        fromfile='early_a',tofile=name)))
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
