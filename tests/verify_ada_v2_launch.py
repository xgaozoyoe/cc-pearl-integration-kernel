"""Two full grids under a CUDA launch trace; identity check, not a benchmark."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ.setdefault('PEARL_TRIAL_LIBRARY','/tmp/libpearl-ada-v2-retest.so')
import hashlib,json,sys
from pathlib import Path
variant=int(sys.argv[1]) if len(sys.argv)>1 else 104
assert variant in (104,109,110,117,120,121)
names=[str(i) for i in range(122)]
names[variant]='hand Ada v2 launch verification'
os.environ['PEARL_TRIAL_NAMES']=json.dumps(names)
import bench_cute_grid as grid

class Scheduler(grid.Scheduler):
    def maybe_switch(self):
        self.grids+=1
        return self.grids>=2

grid.Scheduler=Scheduler
path=Path(grid.lib._name).resolve()
print(json.dumps({'loaded_trial_library':str(path),
                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'variant':variant}),flush=True)
result=grid.run(variant)
assert len(result['times'])==2
print('Two full grids completed; use Nsight kernel names to verify dispatch.',flush=True)
