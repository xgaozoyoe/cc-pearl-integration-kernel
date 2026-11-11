"""Reproducible static checks and summaries for stage/small-tile controls."""
import json,re,math
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];art=root/'artifacts'
results={}
for prefix,names in [('stage-rotation',['fixed','pingpong','global','full']),('small-current',['baseline','tile128','tile64'])]:
 s=(art/f'rank128-{prefix}.sass').read_text();namespace=prefix.replace('-','_');stat={}
 for name in names:
  sec=next(x for x in s.split('Function : ')[1:] if namespace+'_'+name+'6kernel' in x.splitlines()[0])
  rows=re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)
  stat[name]={op:sum(ins.startswith(op) for pc,ins in rows) for op in ('IMMA','LDSM','LDGSTS','LDL','STL','BAR.SYNC','DEPBAR')}
  assert stat[name]['LDL']==stat[name]['STL']==0
 (art/f'rank128-{prefix}-static.json').write_text(json.dumps(stat,indent=2)+'\n');results[prefix]={'static':stat}
 for kind in ('events','results'):
  p=art/f'rank128-{prefix}-{kind}.json'
  if not p.exists():continue
  source=json.loads(p.read_text());stats={}
  for v,name in enumerate(names):
   a=np.array([t for row in source['measurements'] if row['variant']==v for t in row['samples_ms']]);mean=float(a.mean())
   stats[name]=dict(samples=len(a),mean_ms=mean,median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)))
   if kind=='results':stats[name].update(grids_sec=1000/mean,equivalent_ths=2**46/mean/1e9)
  results[prefix][kind]=dict(resources=source['resources'],summaries=stats)
  if prefix=='small-current':
   base=stats['baseline']['mean_ms']
   for name in names[1:]:stats[name].update(time_change_pct=100*(stats[name]['mean_ms']/base-1),throughput_change_pct=100*(base/stats[name]['mean_ms']-1))
geometry=[]
for bm,bn,threads,wm,wn,regs,shared,active in [(256,192,384,64,64,168,69632,1),(128,128,256,32,64,119,36864,2),(64,128,256,32,32,74,26624,3)]:
 warps=threads//32;mma=wm*wn*64//4096;blocks=math.ceil(8192/bm)*math.ceil(8192/bn)
 geometry.append(dict(CTA=[bm,bn,64],warp_tile=[wm,wn],registers=regs,shared_bytes=shared,active_CTAs=active,resident_warps=warps*active,
  warp_IMMA_per_K64=mma,executed_MMA_per_region=blocks*warps*64*mma,
  nominal_copy_bytes_per_executed_MMA=(bm+bn)*64/(warps*mma),
  nominal_copy_bytes_per_useful_MMA=blocks*64*(bm+bn)*64/(2**26)))
results['geometry']=geometry
(art/'rank128-stage-small-summary.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
