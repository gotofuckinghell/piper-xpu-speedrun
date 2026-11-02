# Measurements

All runs: Intel Arc A770 16 GB, PyTorch 2.14.1+xpu, Windows Server 2025,
dataset v33 (17 258 clips of 4-6 s, 25.1 h), bf16 autocast, gradient
checkpointing, segment 2048, length bucketing, 30 OMP threads.

Rate = batch * steps / seconds using the last timing block of each run,
i.e. the steady-state window rather than the warm-up.

| Run | Configuration | Steps in block | Seconds | Replicas/s |
|---|---|---:|---:|---:|
| `baseline_pin_true` | batch 112, seg 2048, bf16, checkpointing | 20 | 92.8 | 24.1 |
| `baseline_drift_check` | same config, foreach off, later in the day | 20 | 99.7 | 22.5 |
| `retest` | same config, foreach on | 20 | 97.5 | 23.0 |
| `post_boot` | same config, right after reboot | 20 | 104.5 | 21.4 |
| `ram_preload` | PIPER_COMPACT=1 PIPER_CACHE_RAM=1 | 20 | 95.4 | 23.5 |
| `dataloader_workers2` | num_workers=2, compact cache | 20 | 102.8 | 21.8 |
| `pin_memory_true` | pin_memory=True after driver crash recovery | 20 | 98.4 | 22.8 |
| `repro_80_steps` | 80 steps, lr 1e-5, log every 10 | 40 | 171.9 | 26.1 |
| `foreach_on` | foreach=True in both AdamW | 20 | 103.4 | 21.7 |

## Alignment stage (maximum_path)

- `mas_legacy_in_step`: median 56.7 ms per step over 15 samples (min 47.0, max 65.6)

## foreach AdamW, ABBA (20-step blocks)

- `abba_on_1.log`: 10 steps / 56.1 s -> 20.0 replicas/s
- `abba_off_1.log`: 10 steps / 59.3 s -> 18.9 replicas/s
- `abba_off_2.log`: 10 steps / 57.0 s -> 19.6 replicas/s
- `abba_on_2.log`: 10 steps / 56.6 s -> 19.8 replicas/s

## Notes

- 20-step windows systematically understate throughput by 15-25 % because
  of XPU kernel autotuning and allocator warm-up. Use >= 60 steps.
- A crashed run (`UR_RESULT_ERROR_UNKNOWN`) leaves level_zero in a state
  where the next run is about twice slower until the driver is reset.
- GPU telemetry during steady state: 100 % utilisation, 2400 MHz,
  104 W, 52/54 C, 14.0 GB peak allocated of 16 GB.
