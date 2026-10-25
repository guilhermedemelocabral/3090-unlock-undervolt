#!/usr/bin/env bash
# degrau_vf.sh TAG LOCK SEGUNDOS OFFSET_GPU0 [OFFSET_GPU1 ...] — um degrau da escada de undervolt.
#
# Aplica o lock de clock (-lgc 210,LOCK) e depois os offsets V/F por GPU (MHz, múltiplos de 15) via vf_offset.py,
# com o power limit que já estiver na placa (PL, padrão 220 W, conferido antes e durante). Roda stress_vf.py em
# todas as GPUs ao mesmo tempo, cada uma num container, e confere bit a bit cada resultado.
#
# Aborta tudo (e zera os offsets) com: Xid novo no kmsg, temperatura >= TMAX, PL diferente do esperado, potência
# acima de PL+15 W por 2 s, ou mais de 0,5 s de freio de hardware nos contadores acumulados do driver.
# Divergência bit a bit derruba só a GPU que errou: o offset dela volta a 0 na hora e as outras seguem.
#
# Variáveis: IMAGE (imagem com PyTorch + pynvml), PL (W), TMAX (°C), OUT (pasta de resultados).
# Exige: docker com --gpus, sudo para nvidia-smi -lgc e para o NVML (vf_offset.py).
set -uo pipefail
TAG=${1:?TAG}; LOCK=${2:?LOCK}; SECS=${3:?SEGUNDOS}; shift 3
OFF=("$@")
D=$(cd "$(dirname "$0")" && pwd)
IMG=${IMAGE:-vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}
PL=${PL:-220}; TMAX=${TMAX:-85}; OVER_W=$((PL + 15))
R=${OUT:-$D/../results/raw/vf}; mkdir -p "$R"
mapfile -t UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
N=${#UUIDS[@]}
[ ${#OFF[@]} = "$N" ] || { echo "uso: degrau_vf.sh TAG LOCK SEGUNDOS + um offset por GPU ($N GPUs)"; exit 1; }
exec > >(tee -a "$R/$TAG.log") 2>&1
echo "[$(date -Is)] degrau $TAG lock=$LOCK offsets=${OFF[*]} ${SECS}s"

bad=$(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader,nounits | awk -F', ' -v pl="$PL" '$2+0 != pl {print $1}')
[ -z "$bad" ] || { echo "GATE: GPU(s) $bad fora de ${PL} W"; exit 2; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "GATE: há processo nas GPUs"; exit 2; }
xids() { journalctl -k -b --no-pager | grep -ciE 'NVRM: Xid|fallen off the bus'; }
xid0=$(xids)

zera() { sudo python3 -I "$D/vf_offset.py" reset | sed 's/^/  /'; }
sets=(); for i in "${!OFF[@]}"; do sets+=("$i=${OFF[$i]}"); done
sudo nvidia-smi -lgc 210,"$LOCK" >/dev/null || { echo "GATE: lock falhou (sem lock o offset vira overclock)"; exit 2; }
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
# Tempo acumulado de freio de hardware (HW Thermal Slowdown + HW Power Braking, µs) por GPU. Use o contador, não o
# bit instantâneo hw_slowdown: numa 3090 Ti o bit aparecia "Active" com o contador parado em 0 µs (leitura espúria).
hw_us() { nvidia-smi -q -d PERFORMANCE | awk '/^GPU 0000/{i++} /(HW Thermal Slowdown|HW Power Braking) +: [0-9]+ us/{s[i-1]+=$(NF-1)} END{for(k=0;k<i;k++) print k, s[k]+0}'; }
declare -A hw0 over zerada; while read -r k v; do hw0[$k]=$v; done < <(hw_us)
t_end=$(( $(date +%s) + SECS + 150 ))
sleep 8
while docker ps --format '{{.Names}}' | grep -q '^vf-gpu'; do
  while IFS=', ' read -r idx w pl t; do
    [ "${pl%.*}" = "$PL" ] || abort "GPU$idx com power limit ${pl} W (esperado ${PL})"
    [ "${t%.*}" -ge "$TMAX" ] && abort "GPU$idx a ${t} °C"
    if [ "${w%.*}" -gt "$OVER_W" ]; then over[$idx]=$(( ${over[$idx]:-0} + 1 )); else over[$idx]=0; fi
    [ "${over[$idx]}" -ge 2 ] && abort "GPU$idx acima de ${OVER_W} W por 2 s (${w} W)"
  done < <(nvidia-smi --query-gpu=index,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits)
  while read -r k v; do [ $((v - ${hw0[$k]:-0})) -gt 500000 ] && abort "GPU$k com $(( (v - ${hw0[$k]:-0}) / 1000 )) ms de freio de hardware"; done < <(hw_us)
  [ "$(xids)" -gt "$xid0" ] && abort "Xid novo no kmsg"
  for i in "${!UUIDS[@]}"; do
    if [ -z "${zerada[$i]:-}" ] && grep -q MISMATCH "$R/$TAG-gpu$i.jsonl" 2>/dev/null; then
      echo "[$(date -Is)] GPU$i divergiu bit a bit: offset dela de volta a 0"; sudo python3 -I "$D/vf_offset.py" set "$i=0" >/dev/null; zerada[$i]=1
    fi
  done
  [ "$(date +%s)" -gt "$t_end" ] && abort "passou do tempo"
  sleep 1
done
kill $LOGGER
falhou=0
for i in "${!UUIDS[@]}"; do
  last=$(grep -E '"done"|MISMATCH|Error|error' "$R/$TAG-gpu$i.jsonl" | tail -1)
  s=$(grep '"mhz_p50"' "$R/$TAG-gpu$i.jsonl" | tail -12 | python3 -I -c "
import json,sys,statistics as st
r=[json.loads(l) for l in sys.stdin]
print(f\"MHz {st.median(x['mhz_p50'] for x in r):.0f} | W {st.median(x['w_p50'] for x in r):.0f} (máx {max(x['w_max'] for x in r):.0f}) | duty {st.median(x['duty'] for x in r):.2f}\" if r else 'sem amostras')")
  echo "GPU$i offset=${OFF[$i]}: ${last:-SEM FIM} | $s"
  echo "$last" | grep -q '"mismatch": 0' || falhou=1
done
[ $falhou = 0 ] && echo "[$(date -Is)] DEGRAU OK" || { echo "[$(date -Is)] DEGRAU COM FALHA (GPUs que divergiram já estão em offset 0)"; exit 4; }
