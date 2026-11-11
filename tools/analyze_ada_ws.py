"""Summarize Nsight base-unit raw CSVs. Never substitute theoretical L2 bytes."""
import argparse,csv,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("directory",type=Path)
p.add_argument("--variants",type=int,nargs="+",default=[0,1,2])
p.add_argument("--useful-mma",type=int,default=16777216);args=p.parse_args()
assert args.useful_mma > 0
metrics={
"kernel_ns":"gpu__time_duration.sum",
"tensor_pct":"sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active",
"l2_util_pct":"lts__throughput.avg.pct_of_peak_sustained_elapsed",
"l2_bytes":"lts__t_bytes.sum",
"barrier_pct":"smsp__warp_issue_stalled_barrier_per_warp_active.pct",
"long_scoreboard_pct":"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
"short_scoreboard_pct":"smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
"math_throttle_pct":"smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
"eligible_warps_per_scheduler":"smsp__warps_eligible.avg.per_cycle_active",
"active_warps_per_scheduler":"smsp__warps_active.avg.per_cycle_active",
"issued_warps_per_cycle":"smsp__issue_active.avg.per_cycle_active",
}
summary={}
for v in args.variants:
    path=args.directory/f"v{v}.raw.csv"
    rows=list(csv.reader(path.open()))
    header_index=next(i for i,row in enumerate(rows) if row and row[0]=="ID" and "Kernel Name" in row)
    header=rows[header_index];units=dict(zip(header,rows[header_index+1]))
    records=[dict(zip(header,row)) for row in rows[header_index+2:] if len(row)==len(header) and row[0].isdigit()]
    assert records,f"No launches in {path}"
    entry={"launches":len(records),"metrics":{},"missing":[]}
    for label,metric in metrics.items():
        if metric not in header:
            entry["missing"].append(metric);continue
        vals=[float(row[metric].replace(",","")) for row in records if row[metric]]
        if not vals:entry["missing"].append(metric);continue
        entry["metrics"][label]={"mean":statistics.mean(vals),"values":vals,"unit":units[metric]}
    if "l2_bytes" in entry["metrics"]:
        assert entry["metrics"]["l2_bytes"]["unit"]=="byte",units["lts__t_bytes.sum"]
        entry["l2_bytes_per_useful_mma"]=entry["metrics"]["l2_bytes"]["mean"]/args.useful_mma
    summary[str(v)]=entry
(args.directory/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
