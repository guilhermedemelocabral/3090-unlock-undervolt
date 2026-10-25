#!/usr/bin/env bash
# apply_profile.sh IDX:PL_W:LOCK_MHZ:OFFSET_MHZ [IDX:PL:LOCK:OFFSET ...] — set power limit, clock lock and V/F offset per GPU.
#
# Order matters: power limit, then the clock lock (-lgc 210,LOCK), and only then the V/F offset. A positive offset
# shifts the whole curve up: with the lock it is an undervolt (same clock, lower voltage); WITHOUT the lock the GPU
# climbs to the top of the shifted curve and the offset becomes an overclock. So if the lock fails, that GPU's offset
# is reset to zero. None of this survives a reboot or a driver reload: run it at boot (systemd), optionally on a timer.
#
# Examples (profiles measured in this repository):
#   serving on all 4 GPUs:              sudo ./apply_profile.sh 0:220:1700:120 1:220:1700:120 2:220:1700:120 3:220:1700:120
#   training on one GPU, others idle:   sudo ./apply_profile.sh 0:450:1950:120
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { sed -n '2,11p' "$0"; exit 1; }
failures=0
for spec in "$@"; do
  IFS=: read -r i pl lock off <<<"$spec"
  if nvidia-smi -i "$i" -pl "$pl" >/dev/null; then echo "GPU$i: power limit ${pl} W"; else echo "GPU$i: power limit FAILED"; failures=$((failures + 1)); fi
  if nvidia-smi -i "$i" -lgc "210,$lock" >/dev/null; then
    echo "GPU$i: clock lock 210-${lock} MHz"
    if python3 -I "$D/vf_offset.py" set "$i=$off" >/dev/null; then echo "GPU$i: V/F offset +${off} MHz"
    else echo "GPU$i: V/F offset FAILED"; failures=$((failures + 1)); fi
  else
    echo "GPU$i: clock lock FAILED; resetting the offset (without the lock it would be an overclock)"
    python3 -I "$D/vf_offset.py" set "$i=0" >/dev/null; failures=$((failures + 1))
  fi
done
python3 -I "$D/vf_offset.py" get
exit $((failures > 0))
