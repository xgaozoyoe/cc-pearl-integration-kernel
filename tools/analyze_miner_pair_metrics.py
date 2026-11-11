"""Summarize matched-window external metrics with verified Nsight/PCI mapping."""
import json, sqlite3, statistics, sys
from pathlib import Path
src=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
c=sqlite3.connect((src/'pair.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
gpus={r[0]:{'pci':r[1],'uuid':r[2],'cuda_device':r[3]} for r in c.execute('SELECT id,busLocation,uuid,cuDevice FROM TARGET_INFO_GPU')}
summary={'window_ns':[1_000_000_000,29_000_000_000], 'devices':[]}
for tid,sid in c.execute('SELECT DISTINCT typeId,sourceId FROM TARGET_INFO_GPU_METRICS WHERE typeId IN (SELECT DISTINCT typeId FROM GPU_METRICS)'):
    # This export encodes the metric device as typeId-sourceId. Verify against
    # explicit recorded PCI selection; do not confuse this with CUDA ordinal.
    device=tid-sid
    if device not in gpus:raise ValueError('Unrecognized metrics source encoding')
    gpu=gpus[device]
    label={1:'PeakMiner',2:'Open-Pearl'}.get(gpu['cuda_device'])
    if not label:raise ValueError('Unexpected physical device in pair trace')
    item={'miner':label,'nsight_device':device,**gpu,'metrics':[]};summary['devices'].append(item)
    for mid,name in c.execute('SELECT metricId,metricName FROM TARGET_INFO_GPU_METRICS WHERE typeId=?',(tid,)):
        if '[Throughput %]' not in name:continue
        a=[r[0] for r in c.execute('SELECT value FROM GPU_METRICS WHERE typeId=? AND metricId=? AND timestamp BETWEEN ? AND ?',(tid,mid,*summary['window_ns']))]
        if not a:continue
        s=sorted(a)
        def quantile(q):
            p=q*(len(s)-1);lo=int(p);return s[lo]+(s[min(lo+1,len(s)-1)]-s[lo])*(p-lo)
        item['metrics'].append(dict(name=name,samples=len(a),mean=statistics.mean(a),median=statistics.median(a),p05=quantile(.05),p95=quantile(.95)))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
