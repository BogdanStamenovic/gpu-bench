# gpu-bench

A health check for a used NVIDIA GPU: VRAM integrity, a compute burn whose
every result is verified, thermals and clocks under 10 minutes of full load,
and the driver's error history. Written to check an RTX 4060 before handing it
to someone else, so they get a card that is known to be sound.

## Run

```sh
./healthcheck.py            # full run, ~17 min
./healthcheck.py --quick    # smoke test, ~1 min
```

Writes `results/<card>-<date>.md` (the readable report), `.json` (everything),
and `.telemetry.csv` (1 s samples). Exit 0 = PASS, 1 = FAIL.

Needs `nvidia-smi`, `nvcc` (builds the two CUDA tools on first run, for sm_86
and sm_89 plus PTX, so the same build runs on a 30-series and a 40-series card)
and `journalctl`. Python stdlib only.

## What it does

| Step | Tool | Fails the card if |
|---|---|---|
| Identity | `nvidia-smi -q` | (recorded: VBIOS, power limits, max clocks, PCIe link) |
| Error history | kernel journal | Xid errors during the run; history is reported |
| VRAM integrity | `src/memtest.cu` | any bad word in ~7 GB over 300 passes x 9 patterns (constants, checkerboards, address-in-address, hashed random, moving inversions) |
| Verified burn | `src/burn.cu` | any GEMM result differs from the reference (5 min FP32 SGEMM on CUDA cores, 5 min FP16 GEMM on tensor cores) |
| Thermals/clocks | `nvidia-smi` at 1 s | hardware thermal slowdown or power braking; PCIe link narrower than the card supports |

Warnings (not failures): software thermal slowdown, peak temperature at or
above 87 °C, throughput falling more than 15% over a burn, PCIe replays, AER
messages, a PCIe generation below the card's maximum under load (often the slot).

Both CUDA tools have `--self-test`: memtest flips one bit and must report
exactly one error; burn perturbs one reference element and must report one
mismatch per check. Run these if you change either tool.

### Why some choices look odd

- **Power comes from `nvidia-smi -q -d POWER`'s "Power Samples" block.** On
  driver 615.78 with this 4060, the `power.draw` query fields read `N/A` even at
  full load, while the sample window is populated.
- **The burn uses `CUBLAS_PEDANTIC_MATH` for FP32**, so cuBLAS cannot quietly
  use TF32 tensor cores and the FP32 phase actually loads the CUDA cores.
- **A call watchdog.** On archserver the same GPU serves phone calls
  (hotline-call, gpu-share's call lease). Each GPU-heavy step is stopped within
  ~1 s of a call starting and retried after it ends. Elsewhere those probes
  find nothing and the check runs normally.

## Limitations

- It does not exercise the graphics pipeline (rasterizer, RT cores), the
  display outputs or NVENC/NVDEC. The test machine is headless. A game or a
  3D benchmark on the buyer's machine covers those.
- History is only as long as the journal. On archserver that was 2 boots
  (since 2026-11-03), not the card's life.
- No memory-junction or hotspot temperature: GeForce boards do not expose them
  through nvidia-smi.
- TFLOPS figures are a health screen (a sick card runs slow or throttles), not
  a tuned benchmark. FP32 SGEMM reaches about half the marketing FP32 number
  on Ada, because that number counts dual-issue.
- A PASS is evidence of health at this load for this duration, not a guarantee.
