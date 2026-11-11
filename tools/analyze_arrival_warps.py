"""Analyze existing source-instrumented timestamps; no GPU launches."""
from pathlib import Path
import json
import re
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root / 'artifacts/arrival-order'
result = {'scope': 'Source trace only, not frozen fastest cubin', 'iterations': [], 'sass': {}}
for kt in (0, 1, 16, 17, 30, 31, 46, 47, 62, 63):
    raw = np.concatenate([np.load(out / f'trace-k{kt}-r{r}.npy') for r in range(3)]).astype(np.int64)
    ts = (raw - raw[:, 0, 0, None, None] + 2**31) % 2**32 - 2**31
    early, late = ts[:, :, 0], ts[:, :, 1]
    strips = [early[:, i:i+4].max(axis=1) for i in (0, 4, 8)]
    row = {
        'kt': kt,
        'ctas': len(raw),
        'median_late_minus_early_per_warp': np.median(late-early, axis=0).tolist(),
        'g0_last_early_in_warps_0_to_3_pct': float(100*np.mean(early[:, :8].argmax(axis=1) < 4)),
        'g0_last_late_in_warps_0_to_3_pct': float(100*np.mean(late[:, :8].argmax(axis=1) < 4)),
        'median_strip0_last_early_minus_strip1': float(np.median(strips[0]-strips[1])),
        'median_strip1_last_early_minus_strip2': float(np.median(strips[1]-strips[2])),
    }
    # Fixed-timeline arrival-count model, NOT measured hardware wait time.
    # All-publication readiness is a conservative whole-stage requirement;
    # it is not a proof of the minimal per-warp data dependency set.
    row['arrival_count_model'] = {}
    all_ready = early.max(axis=1)
    for group, own, peer in [('G0', slice(0, 8), slice(8, 12)),
                             ('G1', slice(8, 12), slice(0, 8))]:
        own_late = late[:, own]
        release = np.maximum(own_late.max(axis=1), early[:, peer].max(axis=1))
        wait = release[:, None] - own_late
        ready_wait = np.maximum(all_ready[:, None] - own_late, 0)
        extra = wait - ready_wait
        assert np.all(extra >= 0)
        row['arrival_count_model'][group] = {
            'mean_cycles_to_count_threshold': float(wait.mean()),
            'mean_cycles_to_all_publications': float(ready_wait.mean()),
            'mean_extra_cycles_from_own_late_arrivals': float(extra.mean()),
            'warp_samples_extra_over64_pct': float(100*np.mean(extra > 64)),
        }
    result['iterations'].append(row)
for name in ('source_control', 'trace'):
    body = (out / f'{name}.sass').read_text().split('Function : ')[1]
    mma = 0
    sites = []
    for line in body.splitlines():
        mma += int('IMMA.' in line)
        if any(s in line for s in ('DEPBAR.LE', 'SR_CLOCKLO', 'BAR.ARV', 'BAR.SYNC')):
            sites.append({'mma_instructions_before': mma, 'instruction': line.strip().split(';')[0].strip()})
    result['sass'][name] = sites
(out / 'warp-analysis.json').write_text(json.dumps(result, indent=2)+'\n')
for row in result['iterations']:
    print(row)
