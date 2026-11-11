"""Compare main-thread stalls with real full-size proofs; no GPU or pool use."""
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import pearl_host
from proof_process import Snapshot


def main():
    snapshot = Snapshot(131072, 131072, 4096)
    try:
        snapshot.A.fill(17)
        snapshot.Bt.fill(-23)
        results = []
        expected = None
        for mode in ("thread", "process", "process", "thread"):
            result = []
            errors = []
            done = threading.Event()

            def build():
                try:
                    if mode == "thread":
                        result.append(pearl_host.build_proof_bt(snapshot.A, snapshot.Bt,
                                      16, 48, bytes(32), 256).to_base64())
                    else:
                        result.append(snapshot.build(16, 48, bytes(32), 256, bytes(80), False)["proof"])
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    done.set()

            gaps = []
            start = last = time.perf_counter()
            worker = threading.Thread(target=build)
            worker.start()
            while True:
                finished = done.wait(0.005)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now
                if finished:
                    break
            worker.join()
            if errors:
                raise errors[0]
            if expected is None:
                expected = result[0]
            assert result[0] == expected, "full-size serialized proof differs"
            row = dict(mode=mode, wall_seconds=last-start, main_max_gap_ms=max(gaps)*1000,
                       heartbeat_count=len(gaps), bit_exact=True)
            results.append(row)
            print(json.dumps(row), flush=True)
        Path("artifacts/proof-isolation-results.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        snapshot.close()


if __name__ == "__main__":
    main()
