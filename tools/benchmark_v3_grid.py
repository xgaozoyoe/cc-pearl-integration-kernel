"""Measure full production grids on GPU 0 without submitting shares."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import miner_capi as miner
from pool_common import real_config


class Pool:
    def check_newer_job(self, job_id):
        return None


class Scheduler:
    mode = 'user'
    grids = 0
    def note(self, elapsed):
        pass
    def maybe_switch(self):
        self.grids += 1
        return self.grids >= 6


if __name__ == '__main__':
    # Zero target suppresses hits, isolating GPU work and normal host overhead.
    miner.mine_job(Pool(), real_config(), bytes(80), 0, 'offline-benchmark',
                   4096, 0, Scheduler(), miner.Bufs(4096),
                   {'user': 0, 'dev': 0}, None,
                   lambda message: print(message, flush=True))
