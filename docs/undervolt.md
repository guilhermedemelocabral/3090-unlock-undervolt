# Undervolting RTX 3090 / 3090 Ti on Linux: the process

This is how the undervolt in this repository was built, step by step, and what each step taught us. It uses only the **documented NVML API** and `nvidia-smi`, so it works on a headless Linux box with no X server and no undocumented driver calls.

## 1. What an "undervolt" is on Linux

GeForce cards have no voltage control on Linux. What NVML exposes is a **clock offset on the whole voltage/frequency (V/F) curve**:

- `nvmlDeviceSetGpcClkVfOffset(gpu, +X)` shifts every point of the curve up by X MHz. At any given voltage, the card now runs X MHz faster.
- `nvidia-smi -lgc 210,LOCK` caps the clock at LOCK MHz.

Put together, the card runs at LOCK MHz but picks the voltage that the **stock** curve uses at **LOCK − X** MHz. That is the undervolt:

```
stock curve:      ... 1580 MHz @ V1 ... 1700 MHz @ V2 ...
offset +120:      ... 1700 MHz @ V1 ...                    (whole curve shifted up 120 MHz)
lock at 1700:     runs at 1700 MHz with V1 < V2            -> same clock, lower voltage, less power
```

The rules that follow from this:

1. **Never apply the offset without the lock.** Without a cap, the card climbs to the top of the shifted curve and the offset becomes an **overclock** at maximum voltage. Always set the lock first, then the offset. If setting the lock fails, set the offset back to 0. `tools/apply_profile.sh` does exactly this.
2. **Offsets come in 15 MHz bins.** Use multiples of 15: +60, +120, +180, +195 and so on.
3. **Nothing persists.** A reboot or a driver reload resets both the lock and the offset, so they must be reapplied at boot (see step 8).
4. **The memory offset uses different units.** `nvmlDeviceSetMemClkVfOffset` takes 2× the MHz that `nvidia-smi` reports: +500 gave +250 MHz on a 3090.
5. **You cannot see the voltage.** On these cards `nvidia-smi -q -d VOLTAGE` returns nothing. The undervolt shows up only as power, temperature, work per watt and stability.

### Which regime you are in decides whether it helps

The power limit (PL), the clock lock and the V/F curve interact. In practice there are two regimes:

| regime | what happens | does the undervolt help? |
|---|---|---|
| **lock-limited:** the card sits at the lock with power below the PL (LLM decode, most serving here) | same clock at lower voltage | **less power and heat**, not more speed; a higher lock at the same voltage can buy speed |
| **power-limited:** the card hits the PL and the clock drops (dense GEMM, heavy training) | the limiter picks a lower clock | yes, **while the voltage is above the floor**: less voltage per clock means more work per watt. On the 3090s, +240 at 1600 MHz gave 18–22% more GEMM work at the same ~180 W. The 3090 Ti stopped gaining past +120 at 1600 MHz, probably because it was already at its minimum voltage |

## 2. Prerequisites

- **Driver:** recent enough to expose the APIs; tested on 595.99.02 (open). Check the allowed range first: on these cards `nvmlDeviceGetGpcClkMinMaxVfOffset` returned [−1000, +1000] MHz, and memory returned [−2000, +6000].
- **Root:** needed for `nvidia-smi -pl/-lgc` and the NVML setters.
- **Persistence mode:** `nvidia-smi -pm 1`.
- **A bit-exact stress test.** "It didn't crash" is not a valid test (see step 5).
- **A power budget.** Here: 220 W per GPU with four GPUs in load (1250 W PSU), and the limiter raised only when **one** GPU is loaded.

## 3. Baseline

Before touching anything, measure:

```bash
# per-GPU TFLOPS/MHz and power at the current limits
docker run --rm --gpus '"device=0"' -v "$PWD/tools":/bench:ro --entrypoint python3 <image with torch> /bench/bench_gemm.py
# whole-system power and PSU rails, one line per second (root)
sudo python3 -I tools/power_monitor.py --psu-w 1250 --csv baseline.csv
```

Also record the baseline of your **real** workload: tokens/s, s/step, TTFT. This is what the undervolt has to beat or match.

## 4. Pick the power limit and the target lock

- **Power limit:** your safety budget. Treat it as an **average, not a ceiling**: with four GPUs at 220 W under memory-heavy load, 100 ms peaks reached 234–265 W per GPU, and GPUs + CPU hit ~1080 W.
- **Target lock:** where your workload lives.
  - For lock-limited serving, start at the clock you already run (here 1600 MHz) and later try one or two bins higher with the offset.
  - The 3090s here never went above **~1725 MHz** even with a 1800 lock and power to spare (179 W, 64 °C), with every throttle reason in NVML at 0. Our guess is a voltage limit that NVML does not expose. In four-GPU tensor parallel, the slowest card sets the pace, so a lock above ~1725 buys nothing.

## 5. Climb the offset ladder with a bit-exact stress

`tools/vf_step.sh` runs one rung on every GPU at once. It takes a tag, the lock, the duration and one offset per GPU:

```bash
sudo -v
PL=220 ./tools/vf_step.sh L1600-o060 1600 180 60 60 60 60
PL=220 ./tools/vf_step.sh L1600-o120 1600 180 120 120 120 120
# ... +180, +240, then repeat at the next lock you care about (1700, ...)
```

What one rung does:

1. Checks that every GPU is at the expected PL and that no process is using the GPUs.
2. Applies the lock, then the offsets.
3. Runs `tools/stress_vf.py` on every GPU. A closed-loop duty cycle on a BF16 GEMM holds the card **at the lock clock** with as much load as the power limit allows; at 1600 MHz that was ~180 W per GPU. This is the regime the serving workload lives in. Every result is compared **bit for bit** with the first.
4. Watches for:
   - a new Xid in the kernel log;
   - temperature at or above 85 °C;
   - the PL changing;
   - power above PL + 15 W for 2 s;
   - more than 0.5 s of hardware braking in the driver's cumulative counters. Use the counters; the instantaneous "HW Slowdown" bit can be spurious.
5. On a mismatch, only that GPU's offset goes back to 0 and the rest keep going. Any other alarm aborts the run and resets all offsets.

Why a plain stress is not enough: **undervolt instability shows up as wrong arithmetic without a crash.** One 3090 at 1700 MHz / +240 produced a different GEMM result after 16 s, with no Xid and no hang. The card would have kept running and computing garbage.

What we measured (220 W, four GPUs at once, 180 s per rung):

| lock | +60 | +120 | +180 | +240 |
|---|---|---|---|---|
| 1600 MHz | ok | ok | ok | ok |
| 1700 MHz | ok | ok | ok | **one 3090 mismatched** |
| 1800 MHz | 3090 Ti ok up to +180; the 3090s capped at ~1725 MHz | | | |

At 1600 MHz and the same ~180 W, +240 let the 3090s do **18–22% more work** per second. The 3090 Ti stopped improving above +120, probably because it was already at its voltage floor. The 1800 MHz rungs only count for the 3090 Ti: the 3090s never reached the lock, and the stress loop fell to a 5% duty cycle.

## 6. Leave a margin

Take the highest rung that passed on **every** GPU and back off by at least 60 MHz (two or more 15 MHz bins). The frontier differs per chip and shifts with load. Here:

- at 1700 MHz the weakest 3090 failed at +240 and passed +180, so we chose **+120**;
- that gives 1700 MHz with roughly the voltage the stock curve uses at 1580 MHz.

## 7. Validate on the real workload, under the real load

The stress validates a point **at the current it ran with**. Higher current means more voltage droop, which eats the margin. Here:

- **Serving (four GPUs, 220 W):**
  - **Same-session ABA:** 1600/0 → 1700/+120 → 1600/0 on one running inference server. 1700/+120 gave **+2.6–3.8% decode**, 1.1% lower TTFT at 82k and 1–8 W more per GPU at p99. The exact-output check matched in all three arms.
  - **Full battery in a separate server session:** no errors, no Xid, and every exact-output and quality check passed.
    - Speed against the unlocked 1600/0 run was mixed: most numbers within ±2%, but reasoning decode was −8.6% (MTP acceptance 0.667 → 0.630) and one concurrency phase was −9–10%. A second phase didn't reproduce the drop, so we read it as variance between server sessions.
    - The long-context retrieval score went 0.958 → 0.940; the stock run gave 0.944.
  - We promoted it **without** a same-session quality A/B or a 10-minute soak at 1700/+120. Silent errors are exactly the risk an undervolt adds, so a careful setup should run both.
- **Training (one 3090 Ti, PL raised to 450 W):**
  - the card ran 1800/+120, 1900/+60, 1900/+120, 1950/+120, 1950/+180 and 1995/+180 cleanly at ~370–430 W;
  - **2010 / +180 produced two anomalies in ~200 steps: one loss of 6×10³¹ with a normal grad_norm, and one infinite grad_norm.** A corrupted logit fits the first one: cross-entropy explodes, but the softmax gradient is bounded. There were no anomalies before or after;
  - +180 had passed the bit-exact stress at 1800 MHz with ~180 W; at 2010 MHz with ~425 W it did not hold;
  - we stepped back to **1950/+120**: same speed within noise, no anomalies over 178+ steps.

So validate with your real outputs:
- **Training:** watch loss and grad_norm in every logged step. Flag NaN, inf and spikes, and compare the mean against the previous window.
- **Inference:** run exact-output tests and a quality battery, not just a speed run.

Keep the last checkpoint from before any change, since trainers usually keep only the most recent few.

## 8. Persist it safely

Apply at every boot and on a timer, because a driver reload or GPU reset undoes the settings. Order per GPU:

```bash
nvidia-smi -pm 1
nvidia-smi -i $i -pl $PL
if nvidia-smi -i $i -lgc 210,$LOCK; then
  python3 -I vf_offset.py set $i=$OFFSET        # only after the lock succeeded
else
  python3 -I vf_offset.py set $i=0              # no lock -> no offset (it would be an overclock)
fi
```

`tools/apply_profile.sh` does this for a list of `index:PL:LOCK:OFFSET` entries. Run it from a systemd oneshot ordered after the NVIDIA driver and **before** any workload starts.

Also:
- **Pin the driver and kernel.** Keep them out of unattended upgrades and update them by hand, then re-test the ladder.
- **Read the offset back after applying it.** `vf_offset.py get` prints the GPC and memory offsets per GPU.
- **Plan for the API change.** `nvmlDeviceSetGpcClkVfOffset` is marked "will be deprecated" in favour of `nvmlDeviceSetClockOffsets`.

## 9. Profiles we ended up with

| use | power limit | lock | offset | why |
|---|---|---|---|---|
| serving on 4 GPUs | 220 W each | 1700 MHz | +120 | lock-limited regime; +2.6–3.8% decode in the same-session ABA, PSU at ~885 W peak for GPUs + CPU |
| training on one 3090 Ti (others idle) | 450 W | 1950 MHz | +120 | **~2.3×** a 260 W no-unlock baseline in tokens/s; 2010/+180 caused silent errors |

## 10. Pitfalls we hit

- **The memory overclock gained nothing.** In CUDA these cards run memory in P2, 250 MHz below P0.
  - Raising it to the P0 clock gave **no bandwidth** under the power cap (694–759 GB/s per GPU either way), only more power. In a pure copy loop on one 3090, bandwidth even fell from ~595 to 572 GB/s, because the copy alone already reached the 220 W limit.
  - +5% above P0 pushed one board to 236 W within 10 s and was aborted.
  - Under memory-heavy load the limit is soft: two 3090s averaged 225–229 W at 220 W.
- **A stress loop can stop exercising the point.** If the duty-cycle controller lowers load whenever the clock is below the lock, a card that can't reach the lock spirals down to idle and the rung tests nothing. Lower load only when the clock dropped **because of power** (fixed in `stress_vf.py`).
- **The HW Slowdown bit can lie.** It flickered "Active" on a 3090 Ti while the cumulative HW Thermal Slowdown and HW Power Braking counters stayed at 0 µs. Use the counters.
- **Removing the lock didn't make training faster.** With +60 and a 450–480 W limit, the free-running 3090 Ti boosted to ~1950 MHz but sat on the power limiter at 440–454 W and 78–80 °C. It ran 13.5 s/step, slower than with a 1900 MHz lock (12.0–13.4 s/step) and hotter. Without a lock, the offset is also an overclock: the top of the curve moved from 2100 to 2160 MHz.

## 11. Alternatives

- **Point-by-point curve editors:** [LACT](https://github.com/ilya-zlobintsev/LACT) (≥ 0.9) and NVCurve edit the V/F curve like MSI Afterburner. They use undocumented NvAPI calls that can break with any driver update.
- **Offset + lock (this guide):** reaches the same "flatten above a point" result with a documented API. What you give up is picking an exact millivolt value.
