"""GPU-0 bit-exact setup_job parity against the accepted CPU V3 reference.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/test_v3_gpu_seeds.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
import cuda_capi as cc
import pearl_host


def check(m, n, k, seed, key):
    arrays = [np.empty((m, k), np.int8), np.empty((n, k), np.int8)]
    buffers = [cc.DBuf(size) for size in (m*k, k*n, n*k, 32, 32, 32)]
    a, b, bt, job, ns_a, ns_b = buffers
    try:
        job.from_host(np.frombuffer(key, np.uint8))
        start = time.perf_counter()
        cc.setup_job(a, b, bt, job, ns_a, ns_b, m, n, k, 256, seed)
        cc.sync()
        gpu_seconds = time.perf_counter() - start
        a.to_host(arrays[0]); bt.to_host(arrays[1])
        actual = []
        for buf in (ns_a, ns_b):
            out = np.empty(32, np.uint8)
            buf.to_host(out)
            actual.append(out.tobytes())
        start = time.perf_counter()
        expected = pearl_host.commitment_hashes_bt_v3(*arrays, key)
        cpu_seconds = time.perf_counter() - start
        assert tuple(actual) == expected, (m, n, k, [x.hex() for x in actual], [x.hex() for x in expected])
        print(f'PASS M={m} N={n} K={k} seed={seed}: GPU setup {gpu_seconds:.4f}s; CPU reference {cpu_seconds:.4f}s', flush=True)
    finally:
        for buf in buffers:
            buf.free()


if __name__ == '__main__':
    # Unequal row counts catch swapped dimensions; cover multiple Merkle levels.
    for shape, seed, key in [((256, 512, 1024), 0, bytes(32)),
                             ((512, 256, 2048), 123456789, bytes(range(32))),
                             ((768, 1280, 1024), 2**64-2, bytes(reversed(range(32))))]:
        check(*shape, seed, key)
    from pool_common import M, N, K
    check(M, N, K, 42, bytes(range(32)))
