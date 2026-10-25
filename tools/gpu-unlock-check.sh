#!/usr/bin/env bash
# gpu-unlock-check.sh — checks that the tensor unlock (gpu-unlock) is still in effect on this boot, so it cannot vanish
# silently. The module has no runtime status; the signals are the kernel log and the events that undo the unlock:
#   1. every PCI address in gpu_unlock.targets= on the kernel command line has "gpu_unlock: unlocked" in this boot's
#      kernel log (missing after a warm reboot, a wrong load order or a refused card: only a real power-off fixes it);
#   2. the nvidia driver loaded only once and there was no Xid / fall off the bus: a reset or reload puts the factory
#      selectors back (the boot log still says "unlocked").
# Writes the state to /run/gpu-unlock.status (e.g. for the MOTD), logs to the journal (tag gpu-unlock-check) and exits
# with 1 on a problem, so a systemd unit shows up in `systemctl --failed`. Run it from a oneshot unit + timer.
# GPU_UNLOCK_CMDLINE / GPU_UNLOCK_KMSG / GPU_UNLOCK_STATUS override the inputs and output; GPU_UNLOCK_NOLOG=1 skips logging (tests).
set -uo pipefail
STATUS=${GPU_UNLOCK_STATUS:-/run/gpu-unlock.status}
cmdline=$(cat "${GPU_UNLOCK_CMDLINE:-/proc/cmdline}")
if [ -n "${GPU_UNLOCK_KMSG:-}" ]; then kmsg=$(cat "$GPU_UNLOCK_KMSG"); else kmsg=$(journalctl -k -b --no-pager -o cat); fi
now=$(date '+%Y-%m-%d %H:%M')

state() {  # level, message
  echo "gpu-unlock [$now] $1: $2" > "$STATUS"
  if [ -n "${GPU_UNLOCK_NOLOG:-}" ]; then :
  elif [ "$1" = OK ] || [ "$1" = INFO ]; then logger -t gpu-unlock-check "$1: $2"; else logger -p user.err -t gpu-unlock-check "$1: $2"; fi
  echo "$1: $2"
}

targets=$(grep -oE 'gpu_unlock\.targets=[^ ]+' <<<"$cmdline" | cut -d= -f2)
if [ -z "$targets" ]; then state INFO "not requested (no gpu_unlock.targets on the kernel command line)"; exit 0; fi
IFS=, read -ra T <<<"$targets"
missing=()
for t in "${T[@]}"; do
  grep -qiE "$t: gpu_unlock: unlocked" <<<"$kmsg" || missing+=("$t")
done
loads=$(grep -c 'NVRM: loading NVIDIA UNIX' <<<"$kmsg")
xids=$(grep -ciE 'NVRM: Xid|fallen off the bus' <<<"$kmsg")

if [ ${#missing[@]} -gt 0 ]; then
  state ERROR "$(( ${#T[@]} - ${#missing[@]} ))/${#T[@]} unlocked; missing ${missing[*]}. Power off for real (poweroff, not reboot) and check again"
  exit 1
fi
if [ "$loads" -gt 1 ] || [ "$xids" -gt 0 ]; then
  state WARNING "${#T[@]}/${#T[@]} unlocked at boot, but then the driver loaded ${loads}x and there were ${xids} Xid: a reset may have undone the unlock. Check with bench_gemm.py (BF16 ~0.08 TFLOPS/MHz = ok; ~0.042 = locked) or power off"
  exit 1
fi
state OK "${#T[@]}/${#T[@]} GPUs unlocked on this boot; driver loaded once; no Xid"
