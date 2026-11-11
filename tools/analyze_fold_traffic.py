"""Normalize NCU counters per 16x16-tile R-boundary transcript update."""
import csv,json,statistics,sys,re
from pathlib import Path
UPDATES=(4096//16)*(4096//16)*(4096//256)
BYTE_SCALE={'byte':1,'Kbyte':1000,'Mbyte':1e6,'Gbyte':1e9}
TIME_SCALE_US={'ns':.001,'us':1,'ms':1000,'s':1e6,'nsecond':.001,'usecond':1,'msecond':1000,'second':1e6}

def read(path):
    with path.open() as f: rows=list(csv.DictReader(f))
    units=next(r for r in rows if not r['ID'])
    samples=[r for r in rows if r['ID']]
    if len(samples)!=3: raise ValueError(f'{path}: expected 3 samples, got {len(samples)}')
    result={}
    script=Path(__file__).with_name('profile_fold_traffic.sh').read_text()
    selected=set(re.search("METRICS='([^']+)'",script)[1].split(','))
    for key,unit in units.items():
        if key not in selected:continue
        values=[float(r[key].replace(',','')) for r in samples]
        med=statistics.median(values)
        item={'median':med,'unit':unit,'samples':values}
        if key.endswith('.sum') and unit in BYTE_SCALE:
            item['bytes_per_update']=med*BYTE_SCALE[unit]/UPDATES
        if key.startswith('l1tex__t_sectors_pipe_lsu_mem_local'):
            item['sector_bytes_per_update']=med*32/UPDATES
        if key=='gpu__time_duration.sum':item['microseconds']=med*TIME_SCALE_US[unit]
        result[key]=item
    return result

if __name__=='__main__':
    root=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/pearl-fold-traffic')
    results={'updates_per_region':UPDATES,'profiles':{}}
    for cache in ('cold','warm'):
        for variant in (0,11):
            name=f'{cache}-v{variant}'
            results['profiles'][name]=read(root/(name+'.csv'))
    output=Path(sys.argv[2] if len(sys.argv)>2 else '/tmp/pearl-fold-traffic-analysis.json')
    output.write_text(json.dumps(results,indent=2))
    for cache in ('cold','warm'):
        print(f'\n{cache.upper()} CACHE — baseline vs register transcript BK64')
        for metric,item in results['profiles'][f'{cache}-v0'].items():
            other=results['profiles'][f'{cache}-v11'][metric]
            field='bytes_per_update' if 'bytes_per_update' in item else 'median'
            unit='B/update' if field=='bytes_per_update' else item['unit']
            print(f'{metric}: {item[field]:.6g} -> {other[field]:.6g} {unit}')
