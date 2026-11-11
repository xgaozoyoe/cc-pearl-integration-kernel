# Instruction-level stall attribution

Built /tmp/libpearl-stall-attribution.so with -lineinfo in an independent experiment library. SASS encoding hashes for variants88/89 exactly match the previously benchmarked kernels; debug source information did not change executable instructions. Production library unchanged. Evidence: instruction-stall-binary-check.json.

Run sudo bash tools/profile_instruction_stalls.sh. GPU0 only, variants88/89,32 warmups and3 measured launches, application replay, no cache flush, requested base clocks. Collect SourceCounters PC sampling plus SchedulerStatistics and WarpStateStatistics. Export both individual all/not-issued stall reasons at SASS PCs and correlated CUDA/SASS, plus aggregate counters. Additional passes/instrumentation make this a diagnostic run, not an end-to-end benchmark. Hardware counter access on this host requires sudo, so the collection has not been executed in this preparation step.

Key sync PCs (offsets from kernel entry, both variants):

- 0x0fa0: DEPBAR.LE SB0,0 — wait_group completion check.
- 0x0fb0: BAR.SYNC.DEFER_BLOCKING — main-loop CTA synchronization.
- 0x2990 (88) /0x2960 (89): BAR.SYNC.DEFER_BLOCKING — transcript shared-memory epilogue synchronization.
- LDGDEPBAR at0x0910,0x0f60 and0x15a0 (88)/0x1590 (89) commits asynchronous load groups.

Trace sampled consumer PCs back to preceding LDGSTS/LDSM and their destination registers; a sampled stall PC is where execution is blocked, not automatically the producer responsible for latency. Distinguish main-loop barrier from transcript epilogue before deciding what to optimize. Compare scheduler eligible/issued warps: stalls matter when issue opportunities are missing.

Outputs in /tmp/pearl-instruction-stalls: warm-v{88,89}.ncu-rep, -sass.csv, -cuda-sass.csv, -raw.csv and.log. Analyze CSV with .venv/bin/python tools/analyze_instruction_stalls.py /tmp/pearl-instruction-stalls/warm-v88-sass.csv /tmp/pearl-instruction-stalls/warm-v89-sass.csv --output artifacts/instruction-stall-analysis.json. Parser retains individual PCs, groups by opcode, and computes shares separately for each sampled metric; it never sums overlapping all/not-issued columns into wall-clock percentages.

Optional separate cold-cache experiment: sudo env CACHE_MODES=cold PROFILE_OUT=/tmp/pearl-instruction-stalls-cold bash tools/profile_instruction_stalls.sh. Do not mix cold and warmed samples in one denominator. Neither experiment captures all full-grid behavior.

Preparation validation: shell syntax, Python compile, actual source-page CSV parsing, and SASS binary identity passed. No kernel redesign, proof/seed/target change, or live mining during preparation.
