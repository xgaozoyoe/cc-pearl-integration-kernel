"""GPU0 attribute query and algebraic K128 buffering model; no kernels."""
from pathlib import Path
import ctypes as C,json
root=Path(__file__).resolve().parents[1];out=root/'artifacts/k128-cost-model';out.mkdir(exist_ok=True)
drv=C.CDLL('libcuda.so.1');assert drv.cuInit(0)==0
props={}
for key,id in [('sm_major',75),('sm_minor',76),('shared_per_sm',81),('regs_per_sm',82),('shared_optin_per_cta',97)]:
 v=C.c_int();assert drv.cuDeviceGetAttribute(C.byref(v),id,0)==0;props[key]=v.value
assert props['sm_major']==8 and props['sm_minor']==9,props
rows=[]
for m,n,k,stages in [(256,192,64,2),(256,192,128,2),(256,192,128,1),(256,192,64,3),(192,192,128,2),(256,128,128,2),(192,128,128,2)]:
 operands=(m+n)*k*stages;transcript=m*n//256*16*4;total=operands+transcript
 rows.append(dict(M=m,N=n,K=k,stages=stages,operand_bytes=operands,transcript_bytes=transcript,shared_bytes=total,fits=total<=props['shared_optin_per_cta'],K_iterations=4096//k,ideal_operand_bytes_per_MAC=(m+n)/(m*n),extra_operand_traffic_vs_baseline=(m+n)/(m*n)/(448/(256*192))-1))
result=dict(device=props,models=rows)
(out/'model.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
lines=['GPU0 attributes queried from CUDA driver;no kernels or performance measurements.',str(props),'M N K stages operands transcript total fits']
for x in rows:lines.append(' '.join(str(x[k]) for k in ('M','N','K','stages','operand_bytes','transcript_bytes','shared_bytes','fits')))
lines+=['Current shared transcript format:16uint32words per16x16tile,12288B for256x192.', 'Plain K128 doublebuffer256x192 needs126976B,exceeds101376B CTA limit by25600B.', 'Even eliminating shared transcript entirely leaves114688B operands,still exceeds limit.', 'K128 singlebuffer fits69632B,butsafe reuse needs all readers done BEFORE next copies and all copies done BEFORE next reads.', 'Thus simple singlebuffer requires two logical cross-warp handoffs per128 chunk (~64 total atK4096),and forfeits copy/compute overlap.', 'These are protocol events,not measured stall time or SASS instruction counts. Prologue/epilogue may change counts.', 'Two-K64 source loop unrolling does not remove the64physical stage handoffs.', 'Three K64 slots fit98304B;cannot hold currentK128 plus full futureK128 simultaneously.', 'A3slot/asymmetric protocol could overlap a futureK64,butrequires partial publication/reuse sync;no guaranteed2x reduction.', 'Reg-staging an extraK64 operand tile would be28672B (7168 INT32regs) acrossCTA before layout duplication,vs1024nominal free regs at384x168.', 'Register operand staging cannot simply be added without reducing existing live storage;no spill-free claim.', 'K128 doublebuffer192x128 fits88064B;changesCTA/operandreuse:ideal bytes/MAC increases42.86% vs256x192.', 'A same-size K64 control would be needed for any smallerCTA experiment;not part of this cost-only stage.', 'Historical checked:build_rank128_stages.py isK64 stages2vs3;build_fold_pair_reduce.py pairsK64 iterations but retains innerhandoff.', 'No maintained-fastest K128 doublebuffer implementation found in these checked scripts;not claiming an exhaustive repository history.', 'Conclusion:do not implement naive256x192 K128 doublebuffer;hardware-infeasible. No synchronization speedup claimed.', 'Production libraries and mining path untouched.']
(out/'report.txt').write_text('\n'.join(lines)+'\n')
