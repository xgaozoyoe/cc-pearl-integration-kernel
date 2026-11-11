"""GPU0 guard; optionally allow one verified non-GPU0 miner supervisor context."""
from pathlib import Path
import subprocess,argparse
root=Path(__file__).resolve().parents[1]
def check_gpu0_idle(supervisor_pid=0):
 uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 for row in rows.splitlines():
  gpu,pid=[x.strip() for x in row.split(',')]
  if gpu!=uuid:continue
  if int(pid)!=supervisor_pid:raise SystemExit(f'GPU0 busy: PID {pid}')
  proc=Path('/proc')/pid;args=proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
  assert '--single' not in args and '--devices' in args,'not a supervisor'
  devices=args[args.index('--devices')+1].split(',')
  assert devices and all(x.isdigit() and int(x)!=0 for x in devices),'supervisor may launch GPU0'
  cwd=proc.joinpath('cwd').resolve();script=Path(args[1]);script=script if script.is_absolute() else cwd/script
  assert script.resolve()==root/'packaging/p40_miner_lite_main.py','unexpected supervisor program'
  print(f'Allowing verified supervisor PID {pid}, devices {devices}; no GPU0 mining child.',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--allow-supervisor-pid',type=int,default=0);a=p.parse_args();check_gpu0_idle(a.allow_supervisor_pid)
