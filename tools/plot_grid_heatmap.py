"""Plot GPU stage occupancy from the isolated variant-20 Nsight trace."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source = Path('/tmp/pearl-best-profile')
out = Path(__file__).resolve().parents[1] / 'artifacts'
c = sqlite3.connect(source / 'grid.sqlite')
events = c.execute('SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').fetchall()
labels = ['GEMM + transcript fold', 'Noise GEMM', 'Random A/B generation',
          'Merkle roots', 'PoW BLAKE3', 'Matrix transpose',
          'Noise seed expansion', 'V3 bind_roots', 'Other kernels',
          'GPU copies / memset', 'Unattributed wall time']
def category(name):
    for needle, i in [('CommitmentHash',7),('cute_pearl_fold',0),('cute_noise_gemm',1),('fill_rand_i8',2),
                      ('Merkle',3),('ComputeBlakeMT',3),('ReduceRoots',3),
                      ('pearl_blake3',4),('transpose_i8',5),('NoiseGeneration',6),
                      ('CommitmentHash',7)]:
        if needle in name: return i
    return 8
starts = [a for a,b,n in events if 'fill_rand_i8' in n][::2]
assert len(starts) == 6
grids = []
for i,start in enumerate(starts):
    limit = starts[i+1] if i+1 < 6 else float('inf')
    group = [(a,b,category(n)) for a,b,n in events if start <= a < limit]
    end = max(b for a,b,cat in group)
    grids.append((start,end,group))
timing = json.loads((source/'timing.json').read_text())
wall_ms = np.mean(timing['times'][1:])*1000
means = np.zeros(len(labels))
for start,end,group in grids[1:]:
    for a,b,cat in group: means[cat] += (b-a)/1e6/5
    for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
        means[9] += c.execute(f'SELECT COALESCE(SUM(end-start),0) FROM {table} WHERE start>=? AND end<=?',(start,end)).fetchone()[0]/1e6/5
means[-1] = wall_ms-means[:-1].sum()
assert means[-1]>=0
percent = 100*means/wall_ms
# Single representative steady grid, 1 ms bins. Values are occupied fraction
# of each bin; this is a stage timeline, not a spatial SM-utilization map.
start,end,group = grids[2]
length = (end-start)/1e6
bins = int(np.ceil(length))
heat = np.zeros((len(labels)-1,bins))
for a,b,cat in group:
    lo,hi = (a-start)/1e6,(b-start)/1e6
    for j in range(int(lo),min(bins,int(hi)+1)):
        heat[cat,j] += max(0,min(hi,j+1)-max(lo,j))
for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
    for a,b in c.execute(f'SELECT start,end FROM {table} WHERE start>=? AND end<=?',(start,end)):
        lo,hi=(a-start)/1e6,(b-start)/1e6
        for j in range(int(lo),min(bins,int(hi)+1)):
            heat[9,j] += max(0,min(hi,j+1)-max(lo,j))
fig,(ax,ht) = plt.subplots(2,1,figsize=(14,9),gridspec_kw={'height_ratios':[1.2,1]})
fig.subplots_adjust(left=.24,right=.86,top=.88,bottom=.09,hspace=.42)
fig.suptitle('RTX 4090 GPU 0 | optimized 2x4 warp / register transcript / BK64',fontsize=17,y=.96)
fig.text(.24,.91,f'Latest Nsight Systems trace | 5 steady grids | mean wall {wall_ms:.2f} ms | no HIT / proof submission',fontsize=10)
y=np.arange(len(labels))
colors=plt.cm.YlOrRd(np.clip(percent/100,.08,1))
ax.barh(y,percent,color=colors)
ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,105)
ax.set_xlabel('Share of actual grid wall time (%)')
ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
for i,(p,ms) in enumerate(zip(percent,means)):
    ax.text(max(p+.8,1),i,f'{p:.3f}%   {ms:.3f} ms',va='center',fontsize=9)
im=ht.imshow(np.clip(heat,0,1),aspect='auto',cmap='magma',vmin=0,vmax=1,
             extent=[0,bins,len(labels)-1-.5,-.5],interpolation='nearest')
ht.set_yticks(np.arange(len(labels)-1),labels[:-1]); ht.set_xlabel('Time within representative grid (ms)')
ht.set_title('Stage activity heatmap | 1 ms bins | bright = GPU executing this stage',fontsize=11)
cb=fig.colorbar(im,ax=ht,fraction=.025,pad=.02); cb.set_label('Bin occupancy')
fig.savefig(out/'optimized-grid-heatmap.png',dpi=180)
fig.savefig(out/'optimized-grid-heatmap.svg')
data={'version':'isolated variant 20, warp2x4 register BK64 s3','steady_grids':5,
      'mean_wall_ms':wall_ms,'rows':[{'stage':n,'mean_ms':float(ms),'wall_percent':float(p)} for n,ms,p in zip(labels,means,percent)]}
(out/'optimized-grid-breakdown.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
