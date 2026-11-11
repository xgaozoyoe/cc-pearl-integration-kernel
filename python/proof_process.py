"""Linux CPU proof isolation. Share snapshots by fd; never import CUDA here."""
from __future__ import annotations

import json
import mmap
import os
import queue
import subprocess
import sys
import time

import numpy as np


class Snapshot:
    """Own immutable-after-copy A/Bt storage until its proof has completed."""

    def __init__(self, m, n, k):
        self.shape = (m, n, k)
        self.fd = os.memfd_create("pearl-proof", os.MFD_CLOEXEC)
        try:
            os.ftruncate(self.fd, (m + n) * k)
            self.mapping = mmap.mmap(self.fd, (m + n) * k)
        except BaseException:
            os.close(self.fd)
            raise
        self.A = np.ndarray((m, k), np.int8, buffer=self.mapping)
        self.Bt = np.ndarray((n, k), np.int8, buffer=self.mapping, offset=m * k)

    def close(self):
        self.A = self.Bt = None
        self.mapping.close()
        os.close(self.fd)

    def release(self):
        if hasattr(self, "owner"):
            self.owner.available.put_nowait(self)
        else:
            self.close()

    def build(self, row, col, key, rank, header, verify):
        request = dict(shape=self.shape, row=row, col=col, key=key.hex(),
                       rank=rank, header=header.hex(), verify=verify)
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), str(self.fd)],
            input=json.dumps(request), text=True, capture_output=True,
            pass_fds=(self.fd,), timeout=120,
        )
        if result.returncode:
            raise RuntimeError(f"proof process failed: {result.stderr[-2000:]}")
        return json.loads(result.stdout)


class SnapshotPool:
    """Two prefaulted buffers: bounded memory, no fresh 1 GiB allocation per hit."""

    def __init__(self, m, n, k):
        self.available = queue.Queue(maxsize=2)
        self.buffers = []
        try:
            for _ in range(2):
                snapshot = Snapshot(m, n, k)
                self.buffers.append(snapshot)
                snapshot.A.fill(0)
                snapshot.Bt.fill(0)
                snapshot.owner = self
                self.available.put_nowait(snapshot)
        except BaseException:
            for snapshot in self.buffers:
                snapshot.close()
            raise

    def acquire(self):
        return self.available.get()

    def close(self):
        if self.available.qsize() != len(self.buffers):
            raise RuntimeError("cannot close snapshots with proofs in flight")
        for snapshot in self.buffers:
            snapshot.close()
        self.buffers.clear()


def main():
    import pearl_host

    request = json.load(sys.stdin)
    m, n, k = request["shape"]
    with mmap.mmap(int(sys.argv[1]), (m + n) * k, access=mmap.ACCESS_READ) as memory:
        A = np.ndarray((m, k), np.int8, buffer=memory)
        Bt = np.ndarray((n, k), np.int8, buffer=memory, offset=m * k)
        start = time.perf_counter()
        proof = pearl_host.build_proof_bt(A, Bt, request["row"], request["col"],
                                         bytes.fromhex(request["key"]), request["rank"])
        verification = None
        if request["verify"]:
            verification = pearl_host.verify_proof_local(bytes.fromhex(request["header"]), proof)
        result = dict(proof=proof.to_base64(), build_seconds=time.perf_counter() - start,
                      verification=verification)
        del A, Bt
    print(json.dumps(result))


if __name__ == "__main__":
    main()
