"""GPU0-only matched XOR controls; CUDA events exclude host/setup/PoW."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'python'))
p = argparse.ArgumentParser()
p.add_argument('--profile', action='store_true')
p.add_argument('--variant', type=int, choices=(0, 1, 2, 3), default=0)
args = p.parse_args()
sys.path.insert(0,str(root/'tools'))
from check_gpu0_idle import check_gpu0_idle
if not args.profile:check_gpu0_idle(int(os.environ.get('PEARL_IDLE_SUPERVISOR_PID','0')))
import cuda_capi as cc
P, I = C.c_void_p, C.c_int
lib = C.CDLL(str(root / 'artifacts/libpearl-handoff-aligned.so'))
lib.handoff_aligned_fold.argtypes = [I, P, P, I, I, P, P]
ref = C.CDLL(str(root / 'artifacts/libpearl-rank.so'))
ref.rank_fold.argtypes = [I, P, P, I, I, P, P]
runtime = C.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so')
runtime.cudaEventCreate.argtypes = [C.POINTER(P)]
runtime.cudaEventRecord.argtypes = [P, P]
runtime.cudaEventSynchronize.argtypes = [P]
runtime.cudaEventElapsedTime.argtypes = [C.POINTER(C.c_float), P, P]
runtime.cudaEventDestroy.argtypes = [P]
resources = {}
for v in range(4):
    info = np.zeros(5, np.int32)
    cc._chk(lib.handoff_aligned_fold(v, None, None, 8192, 8192, None, info.ctypes.data), 'info')
    resources[v] = info.tolist()
assert all(x[3] == 1 and x[4] == 0 for x in resources.values()), resources
print('resources', resources, flush=True)
m = n = 8192
a, b, t = cc.DBuf(m*4096), cc.DBuf(n*4096), cc.DBuf(m*n//256*64)
rng = np.random.default_rng(42)
for buf in (a, b):
    buf.from_host(rng.integers(-128, 128, (8192,4096), np.int8))
def launch(v):
    cc._chk(lib.handoff_aligned_fold(v, a.ptr, b.ptr, m, n, t.ptr, None), 'fold')
if args.profile:
    for _ in range(40):
        launch(args.variant)
        cc.sync()
    sys.exit(0)
cc._chk(ref.rank_fold(128, a.ptr, b.ptr, m, n, t.ptr, None), 'reference')
expected = np.empty(m*n//256*16, np.uint32)
t.to_host(expected)
for v in range(4):
    t.memset(0xa5)
    launch(v)
    actual = np.empty_like(expected)
    t.to_host(actual)
    assert np.array_equal(actual, expected), ('transcript mismatch', v)
print('All four full transcripts match accepted R128 at 8192x8192', flush=True)
start, end = P(), P()
cc._chk(runtime.cudaEventCreate(C.byref(start)), 'event create')
cc._chk(runtime.cudaEventCreate(C.byref(end)), 'event create')
results = []
warm_until = time.perf_counter()+3
while time.perf_counter()<warm_until:
    for _ in range(16):launch(0)
    cc.sync()
# Six rotated blocks, 40 samples each variant/block, 16 launches per sample.
for repeat in range(6):
    for j in range(4):
        v = (repeat + j) % 4
        for _ in range(100):
            launch(v)
        cc.sync()
        values = []
        for _ in range(40):
            cc._chk(runtime.cudaEventRecord(start, None), 'event record')
            for _ in range(16):
                launch(v)
            cc._chk(runtime.cudaEventRecord(end, None), 'event record')
            cc._chk(runtime.cudaEventSynchronize(end), 'event sync')
            ms = C.c_float()
            cc._chk(runtime.cudaEventElapsedTime(C.byref(ms), start, end), 'event time')
            values.append(ms.value/16)
        row = dict(variant=v, repeat=repeat, mean_ms=float(np.mean(values)),
                   median_ms=float(np.median(values)), p95_ms=float(np.percentile(values,95)), samples_ms=values)
        row['telemetry_end'] = subprocess.check_output(['nvidia-smi','-i','0',
            '--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'], text=True).strip()
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'samples_ms'}), flush=True)
(root / 'artifacts/handoff-aligned-events.json').write_text(json.dumps(dict(resources=resources, measurements=results), indent=2)+'\n')
for event in (start, end):
    cc._chk(runtime.cudaEventDestroy(event), 'event destroy')
for buf in (a,b,t):
    buf.free()
