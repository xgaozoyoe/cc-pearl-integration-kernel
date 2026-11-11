"""Offline model only. No GPU kernel or production changes. Binary byte units explicit."""
from pathlib import Path
import csv,json,math,statistics
root=Path(__file__).resolve().parents[1];p=root/'artifacts/rank128-astream-ablation-profile'
M=N=131072;K=4096;R=128;P=K//R;SM=128;BM=256;BN=192
C=M*N*4;region_C=8192**2*4;dram_Bps=1008e9 # nominal scenario, not measured sustained bandwidth
partial=P*C;grid_ms=307.6583333333334
model={'inputs':dict(M=M,N=N,K=K,rank=R,boundaries=P,sm_count=SM,nominal_bandwidth_scenario_Bps=dram_Bps,baseline_grid_ms=grid_ms),
 'int32_full_matrix_GiB':C/2**30,'all_boundary_partial_TiB':partial/2**40,'all_boundary_write_read_TiB':2*partial/2**40,
 'all_boundary_dram_floor_ms':2*partial/dram_Bps*1000,'region_partial_GiB':P*region_C/2**30,
 'prefix_additions':(P-1)*M*N,'transcript_output_GiB':M*N//256*16*4/2**30,'coarse_split':[]}
for S in (2,4,8,16,32):
 # Producer folds first split locally. Others preserve each local cumulative boundary.
 # Their endpoints are already included; only first-split endpoint is extra.
 q=P-P//S+1;traffic=2*q*C
 model['coarse_split'].append(dict(splits=S,k_per_split=K//S,stored_matrices_optimistic=q,storage_TiB=q*C/2**40,
  minimum_write_read_TiB=traffic/2**40,dram_floor_ms=traffic/dram_Bps*1000,
  required_cache_TBps_for_24ms=traffic/.024/1e12))
model['launch_sensitivity']=[dict(extra_launches_per_region=n,regions=256,launch_us=us,unhidden_full_grid_ms=256*n*us/1000) for n in (1,2) for us in (3,5,10,20)]
model['cache_batch_example']=dict(assumed_available_MiB=48,partial_MiB_per_CTA=BM*BN*4*P/2**20,CTAs_fitting=48*2**20//(BM*BN*4*P),partial_MiB_for_128_CTAs=SM*BM*BN*4*P/2**20)
model['issue']={}
for variant in (0,1,4):
 rows=list(csv.reader((p/f'v{variant}.raw.csv').open()));i=next(i for i,r in enumerate(rows) if 'ID' in r and 'Kernel Name' in r);header=rows[i]
 data=[]
 for row in rows[i+1:]:
  if len(row)!=len(header):continue
  d=dict(zip(header,row))
  try:float(d['gpu__time_duration.sum'].replace(',',''))
  except ValueError:continue
  data.append(d)
 def mean(k):return statistics.mean(float(d[k].replace(',','')) for d in data)
 # Dynamic warp instruction count; do not multiply by32 lanes.
 hdr=None;imma=all_inst=0
 for row in csv.reader((p/f'v{variant}.sass.csv').open()):
  if not row:continue
  if row[0]=='Kernel Name':hdr=None;continue
  if 'Address' in row and 'Source' in row:hdr=row;continue
  if hdr is None or len(row)!=len(hdr):continue
  d=dict(zip(hdr,row))
  try:count=float(d['Instructions Executed'].replace(',',''))
  except ValueError:continue
  all_inst+=count
  if 'IMMA.' in d['Source']:imma+=count
 count=imma/len(data);cycles=mean('gpc__cycles_elapsed.avg');ns=mean('gpu__time_duration.sum')
 # gpc elapsed is a SM-clock proxy, NOT the directly measured sum of SM elapsed cycles.
 rate=count/(SM*cycles)
 useful=8192**2*K//4096
 expected=math.ceil(8192/BM)*math.ceil(8192/BN)*12*64*64
 assert count==expected,(count,expected)
 model['issue'][variant]=dict(launches=len(data),executed_warp_imma_per_launch=count,useful_equivalent_imma=useful,padding_fraction=count/useful-1,
  kernel_ms=ns/1e6,imma_Ginst_per_sec=count/ns,gpc_elapsed_cycles=cycles,effective_gpc_GHz=cycles/ns,
  estimated_imma_per_SM_elapsed_cycle=rate,estimated_imma_per_SMSP_elapsed_cycle=rate/4,
  estimated_SMSP_cycles_per_imma=4/rate,estimated_fraction_dense_issue_roof=rate/.25,
  theoretical_roof_assumption='0.25 warp m16n8k32 per SM cycle =1024 INT8 MAC/SM/cycle',
  useful_kernel_TMAC_sec=8192**2*K/ns/1000,warp_imma_share_of_instructions=imma/all_inst,
  tensor_active_pct=mean('sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active'))
model['notes']=['Global partial materialization only; floors exclude producer compute, prefix additional traffic, final writes, launch, and synchronization.',
 'Non-first partitions cannot combine local XOR with an offset XOR because integer addition carries. Need full local cumulative matrices, recomputation, or a fused on-chip alternative.',
 'Launch times are sensitivity assumptions, not measured empty-kernel latency; host overhead may overlap GPU execution.',
 'Cycle-normalized issue uses gpc cycles proxy. A direct sm__cycles_elapsed.sum capture is needed for exact SM-cycle normalization.',
 'Dense roof is a stated architectural scenario, not a measured sustained roof. Old roof microbenchmark had different conditions.',
 'Within-CTA split-K/shared partials are a separate mapping experiment; DRAM floors do not rule them out.']
(root/'artifacts/ksplit-cost-model.json').write_text(json.dumps(model,indent=2)+'\n')
print(json.dumps(model,indent=2))
