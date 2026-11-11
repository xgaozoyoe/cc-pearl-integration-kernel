"""GPU0 production kernel + deterministic injected hits; NEVER contacts a pool."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from pathlib import Path
import queue
import re
import statistics
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import miner_capi as miner
import cuda_capi as cc
from pool_common import real_config


class Pool:
    def check_newer_job(self, job):
        return None

    def submit(self, job, proof):
        assert len(proof) > 1000
        return {"result": True, "offline_test": True}


class Scheduler:
    mode = "user"
    def __init__(self):
        self.grids = 0
    def note(self, elapsed):
        pass
    def maybe_switch(self):
        self.grids += 1
        return self.grids >= 28


def main():
    original_read = cc.DBuf.to_host
    original_setup = cc.setup_job
    results = []
    for repeat in range(2):
        for mode in ((False, True) if repeat == 0 else (True, False)):
            miner.PROOF_PROCESS = mode
            bufs = miner.Bufs(4096)
            scheduler = Scheduler()
            injected = set()
            times = []
            snapshots = []
            accepted = dict(user=0, dev=0)
            q = queue.Queue(maxsize=3)

            def read(buffer, host):
                result = original_read(buffer, host)
                grid = scheduler.grids + 1
                if buffer is bufs.dfound and grid in (5, 13, 21) and grid not in injected:
                    host[0] = 1
                    injected.add(grid)
                if buffer is bufs.dcoord:
                    host[:] = 0
                return result

            def setup(*args):
                return original_setup(*args[:-1], 202610070000 + scheduler.grids)

            def log(line):
                match = re.search(r"grid wall: ([\d.]+) s", line)
                if match:
                    times.append(float(match[1]))
                match = re.search(r"share snapshot wall: ([\d.]+) s", line)
                if match:
                    snapshots.append(float(match[1]))
                if any(s in line for s in ("HIT", "snapshot", "proof build", "error", "ACCEPTED")):
                    print(line, flush=True)

            cc.DBuf.to_host = read
            cc.setup_job = setup
            worker = threading.Thread(target=miner._proof_worker, args=(q, accepted, log), daemon=True)
            worker.start()
            try:
                miner.mine_job(Pool(), real_config(), bytes(80), 0, "offline", 4096, 0,
                               scheduler, bufs, accepted, q, log)
                q.join()
                assert accepted["user"] == 3, accepted
                values = times[4:]
                assert len(values) == 24
                row = dict(repeat=repeat, mode="process" if mode else "thread",
                           median_ms=statistics.median(values)*1000,
                           mean_ms=statistics.mean(values)*1000, max_ms=max(values)*1000,
                           equivalent_ths=(2**46 / 1e12) / statistics.mean(values),
                           snapshot_seconds=snapshots, offline_proofs=accepted["user"])
                results.append(row)
                print(json.dumps(row), flush=True)
            finally:
                q.put(None)
                q.join()
                worker.join()
                cc.DBuf.to_host = original_read
                cc.setup_job = original_setup
                if bufs.proof_snapshots is not None:
                    bufs.proof_snapshots.close()
                for value in vars(bufs).values():
                    if isinstance(value, cc.DBuf): value.free()
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, cc.DBuf): item.free()
    Path("artifacts/pool-pipeline-results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
