"""Same elapsed-cycle metrics/window for baseline, candidate and external GPU1."""
import json,sqlite3,statistics,sys,re
from pathlib import Path
src=Path(sys.argv[1]);result={'window_ns':[10_000_000_000,30_000_000_000],'captures':{}}
selected={}
def pci(text):
 parts=text.lower().split(':');return ':'.join(parts[-2:])
for line in (src/'selection.txt').read_text().splitlines():
 m=re.search(r'Physical GPU (\d+) PCI (\S+) -> Nsight (\d+)',line)
 if m:selected[int(m[3])]=(int(m[1]),pci(m[2]))
for variant in ('baseline','candidate'):
 c=sqlite3.connect((src/(variant+'.sqlite')).resolve().as_uri()+'?mode=ro',uri=True)
 gpus={r[0]:dict(pci=r[1],uuid=r[2]) for r in c.execute('SELECT id,busLocation,uuid FROM TARGET_INFO_GPU')}
 devices=[]
 for tid,sid in c.execute('SELECT DISTINCT typeId,sourceId FROM TARGET_INFO_GPU_METRICS WHERE typeId IN (SELECT DISTINCT typeId FROM GPU_METRICS)'):
  device=tid-sid
  if device not in selected or device not in gpus:raise ValueError('Unexpected device encoding')
  physical,bus=selected[device]
  if pci(gpus[device]['pci'])!=bus:raise ValueError('PCI mapping mismatch')
  item=dict(physical_gpu=physical,nsight_device=device,**gpus[device],metrics=[]);devices.append(item)
  for mid,name in c.execute('SELECT metricId,metricName FROM TARGET_INFO_GPU_METRICS WHERE typeId=?',(tid,)):
   if '[Throughput %]' not in name:continue
   a=[r[0] for r in c.execute('SELECT value FROM GPU_METRICS WHERE typeId=? AND metricId=? AND timestamp BETWEEN ? AND ?',(tid,mid,*result['window_ns']))]
   if len(a)<10000:raise ValueError('Insufficient stable-window samples')
   item['metrics'].append(dict(name=name,samples=len(a),mean=statistics.mean(a),median=statistics.median(a)))
 result['captures'][variant]=devices
(src/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for variant,devices in result['captures'].items():
 for device in devices:
  print(variant,'physical GPU',device['physical_gpu'])
  for metric in device['metrics']:
   if any(word in metric['name'].lower() for word in ('tensor','issue','active','dram')):print(' ',metric['name'],round(metric['mean'],3))
