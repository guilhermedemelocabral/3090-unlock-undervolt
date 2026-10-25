#!/usr/bin/env bash
# vf_step.sh TAG LOCK SECONDS OFFSET_GPU0 [OFFSET_GPU1 ...] — one step of the undervolt ladder.
#
# Sets the clock lock (-lgc 210,LOCK) and then the per-GPU V/F offsets (MHz, multiples of 15) via vf_offset.py, keeping
# whatever power limit is on the card (PL, default 220 W, checked before and during the run). Runs stress_vf.py on all
# GPUs at once, one container each, and checks every result bit for bit.
#
# Aborts everything (and resets the offsets) on: a new Xid in the kernel log, temperature >= TMAX, a power limit other
# than PL, power above PL+15 W for 2 s, or more than 0.5 s of hardware braking in the driver's cumulative counters.
# A bit-exact mismatch only stops the GPU that failed: its offset goes back to 0 at once and the others keep going.
#
# Variables: IMAGE (image with PyTorch + pynvml), PL (W), TMAX (°C), OUT (results folder).
# Needs: docker with --gpus, sudo for nvidia-smi -lgc and for NVML (vf_offset.py).
set -uo pipefail
TAG=${1:?TAG}; LOCK=${2:?LOCK}; SECS=${3:?SECONDS}; shift 3
OFF=("$@")
D=$(cd "$(dirname "$0")" && pwd)
IMG=${IMAGE:-vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}
PL=${PL:-220}; TMAX=${TMAX:-85}; OVER_W=$((PL + 15))
R=${OUT:-$D/../results/raw/vf}; mkdir -p "$R"
mapfile -t UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
N=${#UUIDS[@]}
[ ${#OFF[@]} = "$N" ] || { echo "usage: vf_step.sh TAG LOCK SECONDS + one offset per GPU ($N GPUs)"; exit 1; }
exec > >(tee -a "$R/$TAG.log") 2>&1
echo "[$(date -Is)] step $TAG lock=$LOCK offsets=${OFF[*]} ${SECS}s"

bad=$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits | awk -F', ' -v pl="$PL" '$2+0 != pl {print $1}')
[ -z "$bad" ] || { echo "GATE: GPU(s) $bad not at ${PL} W"; exit 2; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "GATE: there are processes on the GPUs"; exit 2; }
xids() { journalctl -k -b --no-pager | grep -ciE 'NVRM: Xid|fallen off the bus'; }
xid0=$(xids)

zera() { sudo python3 -I "$D/vf_offset.py" reset | sed 's/^/  /'; }
sets=(); for i in "${!OFF[@]}"; do sets+=("$i=${OFF[$i]}"); done
sudo nvidia-smi -lgc 210,"$LOCK" >/dev/null || { echo "GATE: clock lock failed (without the lock the offset is an overclock)"; exit 2; }
sudo python3 -I "$D/vf_offset.py" set "${sets[@]}" | sed 's/^/  /' || { zera; exit 2; }

nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,clocks.sm,temperature.gpu \
  --format=csv,nounits -lms 200 > "$R/$TAG-power.csv" 2>&1 &
LOGGER=$!
for i in "${!UUIDS[@]}"; do
  docker run --rm --name "vf-gpu$i" --gpus "\"device=${UUIDS[$i]}\"" -v "$D":/bench:ro --entrypoint python3 "$IMG" \
    /bench/stress_vf.py --seconds "$SECS" --lock "$LOCK" > "$R/$TAG-gpu$i.jsonl" 2>&1 &
done

kill_all() { for i in "${!UUIDS[@]}"; do docker kill "vf-gpu$i" >/dev/null 2>&1; done; }
abort() { echo "[$(date -Is)] ABORT: $*"; kill_all; kill $LOGGER; zera; exit 5; }
# Cumulative hardware braking time (HW Thermal Slowdown + HW Power Braking, µs) per GPU. Use the counter, not the
# instantaneous hw_slowdown bit: on a 3090 Ti the bit showed "Active" while the counter stayed at 0 µs (spurious read).
hw_us() { nvidia-smi -q -d PERFORMANCE | awk '/^GPU 0000/{i++} /(HW Thermal Slowdown|HW Power Braking) +: [0-9]+ us/{s[i-1]+=$(NF-1)} END{for(k=0;k<i;k++) print k, s[k]+0}'; }
declare -A hw0 over zerada; while read -r k v; do hw0[$k]=$v; done < <(hw_us)
t_end=$(( $(date +%s) + SECS + 150 ))
sleep 8
while docker ps --format '{{.Names}}' | grep -q '^vf-gpu'; do
  while IFS=', ' read -r idx w pl t; do
    [ "${pl%.*}" = "$PL" ] || abort "GPU$idx power limit is ${pl} W (expected ${PL})"
    [ "${t%.*}" -ge "$TMAX" ] && abort "GPU$idx at ${t} °C"
    if [ "${w%.*}" -gt "$OVER_W" ]; then over[$idx]=$(( ${over[$idx]:-0} + 1 )); else over[$idx]=0; fi
    [ "${over[$idx]}" -ge 2 ] && abort "GPU$idx above ${OVER_W} W for 2 s (${w} W)"
  done < <(nvidia-smi --query-gpu=index,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits)
  while read -r k v; do [ $((v - ${hw0[$k]:-0})) -gt 500000 ] && abort "GPU$k had $(( (v - ${hw0[$k]:-0}) / 1000 )) ms of hardware braking"; done < <(hw_us)
  [ "$(xids)" -gt "$xid0" ] && abort "new Xid in the kernel log"
  for i in "${!UUIDS[@]}"; do
    if [ -z "${zerada[$i]:-}" ] && grep -q MISMATCH "$R/$TAG-gpu$i.jsonl" 2>/dev/null; then
      echo "[$(date -Is)] GPU$i bit-exact mismatch: its offset is back to 0"; sudo python3 -I "$D/vf_offset.py" set "$i=0" >/dev/null; zerada[$i]=1
    fi
  done
  [ "$(date +%s)" -gt "$t_end" ] && abort "timed out"
  sleep 1
done
kill $LOGGER
failed=0
for i in "${!UUIDS[@]}"; do
  last=$(grep -E '"done"|MISMATCH|Error|error' "$R/$TAG-gpu$i.jsonl" | tail -1)
  s=$(grep '"mhz_p50"' "$R/$TAG-gpu$i.jsonl" | tail -12 | python3 -I -c "
import json,sys,statistics as st
r=[json.loads(l) for l in sys.stdin]
print(f\"MHz {st.median(x['mhz_p50'] for x in r):.0f} | W {st.median(x['w_p50'] for x in r):.0f} (max {max(x['w_max'] for x in r):.0f}) | duty {st.median(x['duty'] for x in r):.2f}\" if r else 'no samples')")
  echo "GPU$i offset=${OFF[$i]}: ${last:-NO END} | $s"
  echo "$last" | grep -q '"mismatch": 0' || failed=1
done
[ $failed = 0 ] && echo "[$(date -Is)] STEP OK" || { echo "[$(date -Is)] STEP FAILED (GPUs that mismatched are already at offset 0)"; exit 4; }
