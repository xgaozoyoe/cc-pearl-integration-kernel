"""Compare complete mainnet-sized grids using isolated CTA variants."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys,json,ctypes,re,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from bench_cute_variants import lib, NAMES, P
import cuda_capi as cc
import miner_capi as miner
from pool_common import real_config

class Pool:
    def check_newer_job(self,job_id):return None
class Scheduler:
    mode='user'
    def __init__(self):self.grids=0
    def note(self,elapsed):pass
    def maybe_switch(self):
        self.grids+=1
        return self.grids>=6

def run(v):
    def pow_variant(a,b,m,n,k,r,key,tgt,tb,dig,found,coord,variant):
        cc._chk(lib.trial_pow(v,cc._as(a),cc._as(b),m,n,k,r,cc._as(key),cc._as(tgt),
                             cc._as(tb),cc._as(dig),cc._as(found),cc._as(coord)),'trial_pow')
    cc.pearl_pow_split=pow_variant
    info=(ctypes.c_int*5)()
    cc._chk(lib.trial_info(v,info),'trial_info')
    buffers=miner.Bufs(4096)
    times=[]
    ths_values=[]
    def log(message):
        print(f'[{v}: {NAMES[v]}] {message}',flush=True)
        match=re.search(r'grid wall: ([\d.]+) s',message)
        if match:times.append(float(match[1]))
        ths_match=re.search(r'grid \d+ done:.*\(([\d.]+) TH/s\)',message)
        if ths_match:ths_values.append(float(ths_match[1]))
    try:
        miner.mine_job(Pool(),real_config(),bytes(80),0,'offline-tuning',4096,0,
                       Scheduler(),buffers,{'user':0,'dev':0},None,log)
    finally:
        for value in vars(buffers).values():
            if isinstance(value,cc.DBuf):value.free()
            elif isinstance(value,list):
                for buf in value:buf.free()
    med=statistics.median(times[1:])
    return {'variant':v,'name':NAMES[v],'times':times,'median_seconds_excluding_first':med,
            'grids_per_second':1/med,'effective_THS':70.368744177664/med,
            'reported_search_THS':ths_values,'median_reported_search_THS':statistics.median(ths_values[1:])}

if __name__=='__main__':
    results=json.loads(Path(os.environ.get('PEARL_RESULT_PREFIX','/tmp/pearl-cute')+'-variants-results.json').read_text())
    ranked=sorted(results.items(),key=lambda p:p[1]['median_ms'])
    if len(sys.argv)>1:
        top=[int(v) for v in sys.argv[1:]]
    else:
        top=[int(next(v for v,x in ranked if int(v)!=0)),
             int(next(v for v,x in ranked if int(v) in (1,2,3,4,7,8,9)))]
    # Bracket candidates with baseline to reveal clock / thermal drift.
    measurements=[run(v) for v in [0,*top,0]]
    Path(os.environ.get('PEARL_RESULT_PREFIX','/tmp/pearl-cute')+'-grid-results.json').write_text(json.dumps(measurements,indent=2))
    print(json.dumps(measurements,indent=2),flush=True)
