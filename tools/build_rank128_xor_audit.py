"""Matched inline-PTX serial control for the existing XOR tree experiment."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
s = (root / 'artifacts/rank128-xor-tree-source.cu').read_text()
start = s.index('namespace xor_tree_balanced')
end = s.index('// namespace xor_tree_balanced', start) + len('// namespace xor_tree_balanced')
tree = s[start:end]
old = '''"lop3.b32 p,%1,%2,%3,0x96; "
                        "lop3.b32 q,%4,%5,%6,0x96; "
                        "xor.b32 r,%7,%8; "
                        "lop3.b32 %0,p,q,r,0x96; }"'''
new = '''"lop3.b32 p,%1,%2,%3,0x96; "
                        "lop3.b32 q,p,%4,%5,0x96; "
                        "lop3.b32 r,q,%6,%7,0x96; "
                        "xor.b32 %0,r,%8; }"'''
assert tree.count(old) == 1
serial = tree.replace(old, new).replace('xor_tree_balanced', 'xor_audit_serial')
serial = serial.replace('// Four logical ops, but two dependency levels:3+3+2 ->3.',
                        '// Four serial logical ops; same asm boundary and eight input constraints.')
original = s[s.index('namespace xor_tree_original'):s.index('// namespace xor_tree_original')+len('// namespace xor_tree_original')]
parts = [original.replace('xor_tree_original', 'xor_audit_original'), serial,
         tree.replace('xor_tree_balanced', 'xor_audit_tree')]
out = '#include <cuda_runtime.h>\n#include <cstdint>\n' + '\n'.join(parts)
out += '\nextern "C" int xor_audit_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {\n'
for v, ns in enumerate(('original', 'serial', 'tree')):
    out += f' if(v=={v})return xor_audit_{ns}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);\n'
out += ' return cudaErrorInvalidValue;\n}\n'
path = root / 'artifacts/rank128-xor-audit-source.cu'
path.write_text(out)
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc', '-std=c++20', '-O3', '-lineinfo',
                '-gencode', 'arch=compute_89,code=sm_89', '-Xcompiler', '-fPIC',
                '--ptxas-options=-v', '--expt-relaxed-constexpr', '-shared', str(path),
                '-o', str(root / 'artifacts/libpearl-rank128-xor-audit-experimental.so')], check=True)
