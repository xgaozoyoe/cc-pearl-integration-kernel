#!/usr/bin/env bash
# Hardware counters require sudo on this host. No production binary changes.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE_OUT="${PROFILE_OUT:-/tmp/pearl-fold-traffic}"
mkdir -p "$PROFILE_OUT"
export CUDA_VISIBLE_DEVICES=0
METRICS='gpu__time_duration.sum,dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__sass_data_bytes_mem_shared_op_ld.sum,sm__sass_data_bytes_mem_shared_op_ldsm.sum,sm__sass_data_bytes_mem_shared_op_st.sum,sm__sass_data_bytes_mem_shared_op_ldgsts.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active'
# Cold isolated kernel and application-replayed warm-cache modes are separate.
for CACHE in cold warm; do
  for VARIANT in 0 11; do
    if [ "$CACHE" = cold ]; then
      REPLAY=(--replay-mode kernel --cache-control all)
    else
      REPLAY=(--replay-mode application --cache-control none)
    fi
    REPORT="$PROFILE_OUT/$CACHE-v$VARIANT"
    /usr/local/cuda/bin/ncu --devices 0 --clock-control base \
      "${REPLAY[@]}" --kernel-name 'regex:.*cute_pearl_fold.*' \
      --launch-skip 32 --launch-count 3 --metrics "$METRICS" \
      -o "$REPORT" .venv/bin/python tests/profile_fold_traffic.py "$VARIANT" \
      > "$REPORT.log" 2>&1
    /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page raw --csv > "$REPORT.csv"
  done
done
.venv/bin/python tools/analyze_fold_traffic.py "$PROFILE_OUT"
