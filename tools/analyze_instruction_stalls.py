"""Group source-page PC samples by opcode; preserve per-PC attribution.

Percentages are shares of a particular sample column, never wall-clock cost.
Handles repeated kernel blocks in ncu --page source --print-source sass --csv.
"""
import argparse,csv,json,re
from collections import defaultdict
from pathlib import Path

def analyze(path):
    header=None;base=None;launch=-1;records=[]
    with path.open() as f:
        for row in csv.reader(f):
            if not row: continue
            if row[0]=='Kernel Name':
                header=None;base=None;launch+=1;continue
            if 'Address' in row and 'Source' in row:
                header=row;continue
            if header is None or len(row)!=len(header): continue
            data=dict(zip(header,row))
            try: pc=int(data['Address'],16)
            except ValueError: continue
            if base is None: base=pc
            source=data['Source'].strip()
            match=re.search(r'(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',source)
            if not match: continue
            op=match[1]
            category=next((name for prefix,name in (
                ('LDGSTS','cp_async'),('LDGDEPBAR','cp_async_commit'),
                ('DEPBAR','cp_async_wait'),('BAR.SYNC','cta_barrier'),
                ('LDSM','shared_load'),('IMMA','mma'),('REDUX','transcript_reduce'),
                ('STS','shared_store'),('LDL','spill_load'),('STL','spill_store'))
                if op.startswith(prefix)),'other')
            samples={}
            for key,value in data.items():
                if 'stall' not in key.lower() and 'sample' not in key.lower(): continue
                if '%' in value or '%' in key: continue
                try: samples[key]=float(value.replace(',',''))
                except ValueError: pass
            records.append({'launch':launch,'pc_offset':hex(pc-base),
                            'opcode':op,'category':category,'source':source,'samples':samples})
    totals=defaultdict(float);groups=defaultdict(lambda:defaultdict(float))
    for r in records:
        for key,value in r['samples'].items():
            totals[key]+=value;groups[r['category']][key]+=value
    return {'file':str(path),'launches':launch+1,'sample_columns':dict(totals),
            'category_samples':{k:dict(v) for k,v in groups.items()},
            'category_percent_of_column':{k:{m:100*x/totals[m] if totals[m] else 0
                for m,x in v.items()} for k,v in groups.items()},'pcs':records}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('files',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    results=[analyze(p) for p in args.files]
    args.output.write_text(json.dumps(results,indent=2))
    for result in results:
        print(result['file'], 'launches:',result['launches'],'sample columns:',list(result['sample_columns']))
        if not result['sample_columns']: print('No numeric PC sample columns; inspect the report export.')
