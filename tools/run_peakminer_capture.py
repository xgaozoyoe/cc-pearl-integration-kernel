"""Temporarily replace only the known GPU1 PeakMiner worker with a bounded trace."""
import os, signal, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BIN='/home/xgao/peakminer/peakminer'
CMD=[BIN,'--coin','pearl','--devices','1','-o','pearl-eu2.luckypool.io:3360','-u','prl1pp2fh58veqcg8x6g3msx0hshznkml6l2q9ev5efj5tsjzjcc2fm6q58xfw9','-w','peakminer-gpu1','--log-file','/tmp/peakminer-gpu1.log','--log-append']
def matches():
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            a=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if not a or Path(a[0]).name!='peakminer':continue
            if a[a.index('--devices')+1]=='1' and a[a.index('-w')+1]=='peakminer-gpu1':result.append(int(p.name))
        except (OSError,ValueError,IndexError):continue
    return result
pids=matches()
if not pids:raise SystemExit('Expected GPU1 worker not found; refusing automatic restart sequence.')
print('Stopping exact GPU1 worker PIDs:',pids,flush=True)
for pid in pids:
    try:os.kill(pid,signal.SIGINT)
    except ProcessLookupError:pass
for _ in range(40):
    if not matches():break
    time.sleep(.25)
if matches():raise SystemExit('GPU1 worker did not stop gracefully; no force kill or second launch attempted.')
try:
    subprocess.run(['bash',str(ROOT/'tools/profile_peakminer_gpu1.sh')],cwd=ROOT,check=True,timeout=120)
finally:
    # Profile script checks device occupancy. Recheck before restoring normal mining.
    uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if uuid in apps:
        print('GPU1 still occupied; did not start a duplicate. Check nvidia-smi.',flush=True)
    else:
        env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('CUDA_DEVICE_ORDER',None)
        with open('/tmp/peakminer-gpu1-restored-console.log','ab') as f:
            p=subprocess.Popen(CMD,cwd='/home/xgao/peakminer',env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print('Normal GPU1 mining restored, launcher PID',p.pid,'console /tmp/peakminer-gpu1-restored-console.log',flush=True)
