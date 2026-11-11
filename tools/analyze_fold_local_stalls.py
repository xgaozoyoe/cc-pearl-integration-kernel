"""Locate local-XOR samples and static RAW/WAR candidates, without causal claims."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path,nargs='?');args=p.parse_args()
root=Path(__file__).resolve().parents[1]
sass=root/'artifacts/rank128-fold_local_late1.sass'
sections=[]
for section in sass.read_text().split('Function : ')[1:]:
    name=section.splitlines()[0]
    if 'fold_local_late1' not in name:continue
    rows=[];mma_count=0
    for line in section.splitlines():
        match=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*/\* 0x',line)
        if not match:continue
        pc,source=match.groups()
        if 'IMMA.' in source:mma_count+=1
        rows.append(dict(pc=hex(int(pc,16)),source=source,mma_count=mma_count))
    # Linear instruction order only: cannot resolve control flow/scoreboards.
    mma_writes={};mma_reads={};local=[]
    for idx,row in enumerate(rows):
        source=row['source']
        if 'IMMA.' in source:
            regs=[int(v) for v in re.findall(r'\bR(\d+)\b',source)]
            if len(regs)==4:
                dest,a,b,c=regs
                for reg in [*range(a,a+4),*range(b,b+2),*range(c,c+4)]:mma_reads[reg]=(idx,row['pc'])
                for reg in range(dest,dest+4):mma_writes[reg]=(idx,row['pc'])
        if 'LOP3.LUT' in source and ('0x96' in source or '0x3c' in source) and 56<=row['mma_count']<64:
            regs=[int(v) for v in re.findall(r'\bR(\d+)\b',source)]
            raw=[dict(register=f'R{r}',mma_pc=mma_writes[r][1],instruction_distance=idx-mma_writes[r][0]) for r in regs[1:] if r in mma_writes]
            war=None
            if regs and regs[0] in mma_reads:
                pos,pc=mma_reads[regs[0]];war=dict(register=f'R{regs[0]}',mma_pc=pc,instruction_distance=idx-pos)
            local.append(dict(**row,prior_mma_writers=raw,prior_mma_reader_of_destination=war))
    sections.append(dict(function=name,variant=1 if '_fast' in name else 0,early_local_xor=local))
result=dict(caveat='Static prior MMA use only; no CFG/dynamic scoreboard model. Distances are instruction slots, not cycles. PC samples can be attributed downstream; percentages are shares of a sampling column, never wall-clock fractions.',static=sections)
if args.directory:
    result['variants']=[]
    for variant in json.loads((args.directory/'pc-stalls.json').read_text()):
        by_reason={}
        for reason,total in variant['sample_columns'].items():
            sums={}
            for row in variant['pcs']:
                key=(row['pc_offset'],row['source'])
                sums[key]=sums.get(key,0)+row['samples'].get(reason,0)
            top=sorted(sums.items(),key=lambda item:item[1],reverse=True)[:20]
            by_reason[reason]=[dict(pc=pc,source=src,samples=value,percent_of_column=100*value/total if total else 0) for (pc,src),value in top if value]
        v=1 if Path(variant['file']).name.startswith('v1') else 0
        early={r['pc'] for sec in sections if sec['variant']==v for r in sec['early_local_xor']}
        early_rows=[r for r in variant['pcs'] if r['pc_offset'] in early]
        # Preserve all columns: fixed-latency wait, scoreboard, throttle, etc.
        early_totals={reason:sum(r['samples'].get(reason,0) for r in early_rows) for reason in variant['sample_columns']}
        result['variants'].append(dict(variant=v,file=variant['file'],top_pcs_by_reason=by_reason,early_local_xor_samples=early_totals,early_local_xor_pcs=early_rows))
    destination=args.directory/'fold-local-attribution.json'
else:destination=root/'artifacts/rank128-fold-local-static-dependencies.json'
destination.write_text(json.dumps(result,indent=2)+'\n')
print(destination)
