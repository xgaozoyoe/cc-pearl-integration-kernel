"""Remove volatile only from pure XOR inline PTX; keep CUDA memory/MMA asm intact."""
from pathlib import Path
import subprocess
root = Path(__file__).resolve().parents[1]
s = (root/'artifacts/rank128-xor-audit-source.cu').read_text()
assert s.count('asm volatile("{ .reg .b32 p,q,r; ') == 2
s = s.replace('asm volatile("{ .reg .b32 p,q,r; ', 'asm("{ .reg .b32 p,q,r; ')
s = s.replace('xor_audit_', 'xor_nonvolatile_')
path = root/'artifacts/rank128-xor-nonvolatile-source.cu'
path.write_text(s)
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo',
 '-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v',
 '--expt-relaxed-constexpr','-shared',str(path),'-o',
 str(root/'artifacts/libpearl-rank128-xor-nonvolatile-experimental.so')],check=True)
