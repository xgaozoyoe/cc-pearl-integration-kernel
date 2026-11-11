"""Offline regression tests: real CPU proofs and a local fake Stratum socket."""
import json
import ast
from collections import deque
import os
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import numpy as np
import pearl_host
from proof_process import Snapshot, SnapshotPool
from pool_common import LuckyPool


class PipelineTests(unittest.TestCase):
    def test_wall_rate_counts_pauses(self):
        # Extract the pure accounting class without initializing CUDA in this
        # CPU-only suite. Exercise irregular batches and a long idle interval.
        path = Path(__file__).resolve().parents[1] / "python/miner_capi.py"
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "WorkRate")
        clock = [0.0]
        scope = {"deque": deque, "time": SimpleNamespace(monotonic=lambda: clock[0])}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        rate = scope["WorkRate"]()
        logs = []
        for timestamp, work in [(10, 100e12), (20, 100e12), (40, 0), (80, 100e12)]:
            clock[0] = timestamp
            rate.record(work, logs.append)
        for actual, expected in zip(logs, ["10.00 TH/s", "5.00 TH/s", "1.67 TH/s"]):
            self.assertIn(expected, actual)
        self.assertEqual(len(logs), 3)

    def test_snapshot_pool_reuses_storage(self):
        pool = SnapshotPool(16, 16, 256)
        first, second = pool.acquire(), pool.acquire()
        first.A.fill(42)
        first.release()
        again = pool.acquire()
        self.assertIs(again, first)
        self.assertTrue(np.all(again.A == 42))
        again.release()
        second.release()
        pool.close()

    def test_process_proof_is_bit_exact(self):
        rng = np.random.default_rng(20261007)
        snapshot = Snapshot(64, 80, 4096)
        try:
            snapshot.A[:] = rng.integers(-128, 128, snapshot.A.shape, dtype=np.int8)
            snapshot.Bt[:] = rng.integers(-128, 128, snapshot.Bt.shape, dtype=np.int8)
            key = bytes(range(32))
            expected = pearl_host.build_proof_bt(snapshot.A, snapshot.Bt, 16, 48, key, 256).to_base64()
            actual = snapshot.build(16, 48, key, 256, bytes(80), False)
            self.assertEqual(actual["proof"], expected)
        finally:
            fd = snapshot.fd
            snapshot.close()
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_notify_before_reply_and_unique_request_ids(self):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        pool = LuckyPool("unused", 0, "public-wallet", "test")
        pool.s = client
        requests = []
        errors = []

        def peer():
            try:
                with server.makefile("rb") as stream:
                    for i in range(2):
                        request = json.loads(stream.readline())
                        requests.append(request)
                        messages = [
                            {"id": request["id"] - 1, "result": False},
                            {"method": "mining.notify", "params": {"header": "00" * 80,
                             "target": "01", "job_id": f"new-{i}"}},
                            {"method": "mining.set_difficulty", "params": 123},
                            {"id": request["id"], "result": True, "error": None},
                        ]
                        server.sendall(b"".join(json.dumps(m).encode() + b"\n" for m in messages))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=peer, daemon=True)
        thread.start()
        for i in range(2):
            self.assertTrue(pool.submit("old-job", "proof")["result"])
            self.assertEqual(pool.check_newer_job("old-job")[2], f"new-{i}")
            self.assertIsNone(pool.check_newer_job(f"new-{i}"))
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertNotEqual(requests[0]["id"], requests[1]["id"])
        self.assertEqual(pool.difficulty, 123)

    def test_eof_is_not_a_job_timeout(self):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        pool = LuckyPool("unused", 0, "wallet", "test")
        pool.s = client
        server.close()
        with self.assertRaises(ConnectionError):
            pool.next_job(timeout=1)

    def test_new_job_visible_while_submission_waits(self):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        pool = LuckyPool("unused", 0, "wallet", "test")
        pool.s = client
        replies = []
        worker = threading.Thread(target=lambda: replies.append(pool.submit("old", "proof")), daemon=True)
        worker.start()
        with server.makefile("rb") as stream:
            request = json.loads(stream.readline())
        server.sendall(json.dumps({"method": "mining.notify", "params": {
            "header": "00" * 80, "target": "01", "job_id": "new"}}).encode() + b"\n")
        deadline = time.monotonic() + 2
        while not pool.pending_jobs and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertTrue(worker.is_alive())
        self.assertEqual(pool.check_newer_job("old")[2], "new")
        server.sendall(json.dumps({"id": request["id"], "result": True}).encode() + b"\n")
        worker.join(2)
        self.assertEqual(replies, [{"id": request["id"], "result": True}])


if __name__ == "__main__":
    unittest.main()
