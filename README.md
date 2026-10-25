# 3090-unlock-undervolt

Measurements of three performance levers on **GA102 cards (RTX 3090 / 3090 Ti)**, taken in October 2026 on a machine with four of them:

1. **Tensor unlock** with [gpu-unlock](https://github.com/Zagitta/gpu-unlock), which removes the half-rate BF16/FP16-acc32/TF32 that NVIDIA imposes on GeForce cards.
2. **V/F curve (undervolt):** a clock offset on the V/F curve through NVML, always combined with a clock lock.
3. **Power limit (PL):** how much power each GPU may draw.

**TL;DR**
- **Training on one GPU:** raising the power limit mattered far more than clocks or offsets. On a 3090 Ti the best measured point was a **1950 MHz lock + 120 MHz V/F offset (undervolt) + 450 W**, at **~2.3×** the throughput of an older run without the unlock at 260 W.
- **Tensor unlock:** it doubles BF16 throughput per clock. At a 220 W cap the power limiter eats half of that, leaving +19–26% in dense GEMM and about −6% prefill time for a W4A16 MoE served with vLLM TP4. Decode stays flat.
- **Undervolt:** it saves 15–20 W per GPU at the same clock. One step too far shows up as **silent compute errors**, with no crash and no Xid, so validate with bit-exact checks. The step-by-step process is in [`docs/undervolt.md`](docs/undervolt.md).

## Results

### Training on one GPU (3090 Ti), the others idle

The workload was QLoRA 4-bit fine-tuning (rank 8, sequences up to 16k) of a ~9B dense model, at ~20.1k tokens per step and an effective batch of 8. Speed is wall time between steps, excluding checkpoint pauses. Different settings fall on different sample windows, so expect ±5–10% noise.

The **baseline is an older run on the same card, without the unlock**, at 260 W and a 1605 MHz lock (LoRA BF16, sequences up to 8k, ~35.1k tokens per step). Because tokens per step differ between the two runs, every speedup is computed from **tokens per second**.

| 3090 Ti setting (lock / V/F offset / PL) | s/step | tokens/s | **vs. baseline** | GPU power (median / peak) | temp. | loss anomalies | steps |
|---|---|---|---|---|---|---|---|
| **baseline:** older run, 1605 MHz, 260 W, **no unlock** | 51.4 | ~683 | **1.00×** | — | — | — | 1,258 |
| 1600 / +120 / 220 W, with unlock | 29.15 | ~689 | 1.01× | ~220 W | 58 °C | — | 62 |
| 1800 / +120 / 450 W | 14.54 | ~1,380 | 2.02× | 369 / 425 W | 71 °C | — | 24 |
| 1900 / +60 / 450 W | 11.97 | ~1,680 | 2.46× | 418 / 472 W | 75 °C | — | 35 |
| no lock / +60 / 450–480 W | 13.48 | ~1,490 | 2.18× | 440–454 / 496 W | 78–80 °C | — | 33 |
| 1900 / +120 / 450 W | 13.37 | ~1,500 | 2.20× | 402 / 475 W | 74 °C | — | 131 |
| 1950 / +120 / 450 W | 12.57 | ~1,600 | 2.34× | 427 / 482 W | 75–77 °C | — | 60 |
| 1950 / +180 / 450 W | 12.27 | ~1,640 | 2.40× | 408 / 485 W | 75 °C | — | 60 |
| 1995 / +180 / 450 W | 11.93 | ~1,680 | 2.47× | ~414 W | 76 °C | — | 28 |
| 2010 / +180 / 450 W (at the PL 29% of the time) | 12.47 | ~1,610 | 2.36× | mean 425 W | 78 °C | **2: loss 6×10³¹, grad_norm inf** | 60 |
| **1950 / +120 / 450 W (sweet spot)** | **12.84** | **~1,560** | **2.29×** | 426 / 487 W | 73–76 °C | — | **178** |

- **The unlock alone matched the baseline.** With 40 W less and less than half the clock (220 W at ~720 MHz against 260 W and up to 1605 MHz), the unlocked card reached the same tokens/s.
- **Raising the power limit did the rest.** From 220 to 450 W the card ran ~2× faster. From 1900 to 2010 MHz everything lands in the same 12–13.4 s/step band, which is within the noise. The longest window, 178 steps at the sweet spot, is the most reliable number.
- **The undervolt shows up in power.** At the same clock, +60 → +120 saved ~15 W and +120 → +180 saved ~19 W.
- **2010 / +180 computed wrong without crashing.** There were two anomalous records in ~215 steps and none before or after:
  - a loss of 6×10³¹ with a normal grad_norm, the signature of one corrupted logit (the softmax gradient is bounded);
  - an infinite grad_norm.

  An offset validated in the stress test at ~180 W does not hold at ~425 W: the higher current eats the voltage margin.
- **No lock is worse.** At 78–80 °C GPU Boost cuts the clock, and the extra watts turn into heat.
- **Caveat on the baseline:** the method and data differ between the two runs, so the baseline comparison is approximate. Data: [`results/training_3090ti_speed.csv`](results/training_3090ti_speed.csv).

### Tensor unlock (GEMM, 220 W)

| GPU | BF16 TFLOPS/MHz stock → unlock | net BF16 TFLOPS stock → unlock |
|---|---|---|
| RTX 3090 Ti | 0.043 → 0.084 | 38.8 → 46.2 (**+19%**) |
| RTX 3090 | 0.042 → 0.081 | 44.7 → 55.0 (**+23%**) |
| RTX 3090 | 0.042 → 0.081 | 49.9 → 62.8 (**+26%**) |
| RTX 3090 | 0.042 → 0.081 | 44.0 → 55.1 (**+25%**) |

- **Control:** FP16 with FP16 accumulation, already full rate, stayed at ~0.08 in both states.
- **Why the net gain is smaller:** at 220 W the power limiter drops the clock from ~1000 to ~650 MHz, so doubling per clock does not double TFLOPS.
- **Joint soak:** a 10-minute soak with all four unlocked had no Xid, p99 power of 220 W and a top temperature of 77 °C.
- Data: [`results/gemm_unlock_220w.csv`](results/gemm_unlock_220w.csv) and [`results/joint_soak_unlock_220w.csv`](results/joint_soak_unlock_220w.csv).

### Inference (vLLM, W4A16 MoE, TP4, 4 GPUs at 220 W)

The model is a MoE with ~6B active parameters (Qwen3.8-Flash-Next family, GPTQ W4A16 / Marlin), served by vLLM with TP4 + DCP2, MTP k=3 and FP8 KV. Each arm ran one full battery.

| metric | stock 1600 MHz | unlock 1600 MHz | unlock 1700 / +120 |
|---|---|---|---|
| cold prefill, 84k (n=12, CV ≤ 0.2%) | 52.5 s | 49.5 s (−5.7%) | 48.8 s |
| cold prefill, ~150k | 98.1 s | 91.8 s (−6.4%) | 93.5 s |
| decode, prose / code (1 stream) | 126 / 186 tok/s | 128 / 188 | 130 / 192 |
| 6 concurrent streams | 370 tok/s | 388 | 393 |

- **Decode barely moves:** it is bound by memory bandwidth.
- **Serving never hits 220 W:** it is held by the clock lock instead.
- **Same-session ABA:** 1700/+120 against 1600/0 gave **+2.6–3.8% decode** for up to ~8 W more per GPU.
- Data: [`results/inference_moe_w4a16_tp4.csv`](results/inference_moe_w4a16_tp4.csv).

### Undervolt frontier (bit-exact stress, 4 GPUs at once, 220 W, 180 s per step)

| lock | +60 | +120 | +180 | +240 |
|---|---|---|---|---|
| 1600 MHz | ok | ok | ok | ok |
| 1700 MHz | ok | ok | ok | **one 3090 mismatched bit for bit after 16 s**, no Xid |
| 1800 MHz | only the 3090 Ti went above ~1725 MHz; ok up to +180 | | | |

- **More work at the same power:** at the same clock and power (~180 W), +240 let the 3090s do **18–22% more work**. The 3090 Ti saturates at +120 at 1600 MHz because it is already at its voltage floor.
- **The 3090s top out around 1725 MHz:** with an 1800 MHz lock they stop at ~1725 MHz, with no throttle reason reported by NVML. In multi-GPU tensor parallel a higher lock does not help.
- **Memory overclock:** both the P0 clock and P0 +5% gave no bandwidth under the PL, only more power.
- Data: [`results/undervolt_ladder_220w.csv`](results/undervolt_ladder_220w.csv).

### The PSU (1250 W): the power limit is an average, not a ceiling

| load | per GPU, median / 100 ms peak | GPUs + CPU (max) | estimate with +60 W for the rest |
|---|---|---|---|
| memory stress, 4 GPUs at 220 W | 214–229 / 234–265 W | 1,082 W | ~1,140 W (**~91%**) |
| MoE inference TP4 (1700/+120) | 167–198 / 180–210 W | 885 W | ~945 W (~76%) |
| training on one GPU (1950/+120, 450 W) | 426 / 487 W | ~620 W | ~680 W (~54%) |

- **The +12 V rail** (motherboard Super I/O, 1 Hz) never went below 11.90 V, but that reading cannot see millisecond transients.
- **Rule followed here:** the power limit is lifted only when **one** GPU is under load.
- Data: [`results/psu_1250w.csv`](results/psu_1250w.csv).

## Machine

| | |
|---|---|
| GPUs | 1× RTX 3090 Ti (EVGA) + 3× RTX 3090, GA102, 24 GB each |
| PCIe | 4.0, no NVLink, no P2P: x4 (3090 Ti, CPU lanes), x8 behind the chipset, x8 + x8 (CPU) |
| CPU / board | Ryzen 7 5800X3D / ASUS Pro WS X570-ACE |
| PSU | 1250 W |
| OS / driver | Ubuntu 24.04, kernel 6.8, NVIDIA 595.99.02 (open) |
| Unlock | [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) at `af28cbb` + `dbe10a5` (Ubuntu initramfs loader), `gpu_unlock.resume=0` |

## Method

> **Full undervolt guide:** [`docs/undervolt.md`](docs/undervolt.md) explains why lock + offset is an undervolt and offset alone is an overclock, which regimes benefit, the bit-exact ladder, margins, validating on the real workload, safe persistence, and the pitfalls we hit.

- **Undervolt through the documented NVML API.** We call `nvmlDeviceSetGpcClkVfOffset` (offset in MHz, a multiple of 15) **after** `nvidia-smi -lgc 210,LOCK`. With the lock in place, the GPU runs at LOCK MHz with the voltage the stock curve uses at LOCK − offset. Without the lock, the same offset is an overclock.
- **Memory offset units:** the memory offset API (`nvmlDeviceSetMemClkVfOffset`) takes **2×** the MHz that `nvidia-smi` shows: +500 gave +250 MHz.
- **Stability is checked bit for bit.** "It did not crash" proves nothing, because undervolt errors show up as **wrong results without a crash**.
  - `tools/stress_vf.py` holds the GPU at the lock clock, close to the PL, and compares every GEMM with the first result.
  - `tools/stress_mem.py` does the same with memory copies.
- **Training speed** is wall time between tqdm steps, excluding checkpoint pauses.
- **Counters, not bits:** the instantaneous "HW Slowdown" bit showed active on a 3090 Ti while the driver's cumulative counters stayed at 0 µs of thermal and power braking. Use the counters (`nvidia-smi -q -d PERFORMANCE`).

## Tools (`tools/`)

| file | what it does |
|---|---|
| `bench_gemm.py` | 8192³ GEMM per data type (BF16/FP16/TF32), in TFLOPS per SM MHz; `--soak N` for continuous load |
| `vf_offset.py` | reads and sets per-GPU GPC and memory V/F offsets (NVML, root) |
| `apply_profile.sh` | sets power limit, clock lock and offset per GPU, in that order; resets the offset if the lock fails |
| `stress_vf.py` | stress at the lock clock, close to the PL, with bit-exact checks |
| `vf_step.sh` | one undervolt ladder step on all GPUs, with a watchdog for Xid, temperature, PL, hardware braking and mismatches |
| `stress_mem.py` | GDDR6X bandwidth and integrity, for memory offsets |
| `power_monitor.py` | system power per second: GPUs (NVML, instantaneous), CPU (RAPL), PSU load and +12/+5/+3.3 V rails from the Super I/O. The mapping is for an ASUS NCT6798D; check yours. |
| `gpu-unlock-check.sh` | checks at boot and periodically that every GPU in `gpu_unlock.targets` was unlocked and that the driver did not reload or hit an Xid (a reset undoes the unlock) |

Examples:

```bash
# TFLOPS/MHz baseline for one GPU (≈0.042 locked, ≈0.08 unlocked)
docker run --rm --gpus '"device=0"' -v "$PWD/tools":/bench:ro --entrypoint python3 \
  vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c /bench/bench_gemm.py

# one ladder step: 1700 MHz lock, +120 on 4 GPUs, 180 s, with the power limit already on the cards (PL=220 by default)
sudo -v && PL=220 ./tools/vf_step.sh L1700-o120 1700 180 120 120 120 120

# training profile on one GPU (the others idle)
sudo ./tools/apply_profile.sh 0:450:1950:120
```

## Risks

- **The unlock** loads signed firmware and rewrites GPU selectors before the driver loads. Read the gpu-unlock README. It is lost on a reset or driver reload, and a warm reboot can be refused: power off for real.
- **The undervolt** can compute wrong **without crashing**. Validate with bit-exact checks and on your real workload (loss, grad_norm, outputs), and leave margin. Here +180 passed at 1800 MHz with ~180 W and failed at 2010 MHz with ~425 W.
- **The PSU:** the power limit is an average. 100 ms peaks exceed it, and four GPUs under memory load took a 1250 W PSU to ~90% (estimated).
- **Driver and kernel updates** can break the unlock. Here they were removed from unattended-upgrades and put on `apt-mark hold`.

Use at your own risk.

## Credits

- **[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock):** author of the module that unlocks the tensor rate on GA10x. The unlock tests here exist because of it, and the compatibility reports in its issues (e.g. #15, #25, #75) shaped the expectations.
- **[noonghunna/club-3090](https://github.com/noonghunna/club-3090):** community measurements on 3090s (W4A8 vs W4A16, P2P).
- **[NVIDIA NVML](https://docs.nvidia.com/deploy/nvml-api/):** V/F offset, clock lock and throttle counter APIs.
- **[Henry Freed, "GPU Power Limits and Undervolting on Linux"](https://hbfreed.com/2026/03/12/power-limits-undervolting.html):** lock + offset + PL on 3× 3090 under Linux.
- **Ampere undervolting forums** (Igor's Lab, ComputerBase, TechPowerUp, Overclockers UK, EVGA): V/F reference points for the 3090/3090 Ti.
- **[vLLM](https://github.com/vllm-project/vllm):** the inference engine used in the tests.

## License

The scripts and text in this repository are under the MIT license (see `LICENSE`). gpu-unlock has its own license and is not redistributed here.
