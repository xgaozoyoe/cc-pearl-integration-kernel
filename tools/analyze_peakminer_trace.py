"""Read-only Nsight Systems kernel inventory; no per-grid TH/s inference."""
import json, sqlite3, sys
from pathlib import Path
path = Path(sys.argv[1]).resolve()
con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
    raise SystemExit('No CUDA kernels recorded; profiling injection may not have reached the GPU process.')
cols = {r[1] for r in con.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
name = next((x for x in ('demangledName', 'shortName', 'mangledName') if x in cols), None)
if not name or 'StringIds' not in tables:
    raise SystemExit('Unknown Nsight kernel name schema; inspect SQLite manually.')
fields = [x for x in ('deviceId','contextId','gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory') if x in cols]
select = ','.join('k.'+f for f in fields)
query = ('SELECT s.value, '+select+', COUNT(*), SUM(k.end-k.start), AVG(k.end-k.start) '
         'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.'+name+
         ' GROUP BY s.value,'+select+' ORDER BY SUM(k.end-k.start) DESC')
rows=[]
for row in con.execute(query):
    result={'kernel':row[0], **dict(zip(fields,row[1:1+len(fields)])),
            'launches':row[-3], 'sum_kernel_ms':row[-2]/1e6, 'mean_kernel_ms':row[-1]/1e6}
    rows.append(result)
print(json.dumps({'note':'Summed kernel durations may overlap; not full-grid wall time or useful MMA counts.', 'kernels':rows},indent=2))
