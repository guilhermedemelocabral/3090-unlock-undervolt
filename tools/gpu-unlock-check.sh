#!/usr/bin/env bash
# gpu-unlock-check.sh — confere que o unlock dos tensores (gpu-unlock) continua valendo neste boot, para ele não
# sumir em silêncio. O módulo não tem status em tempo de execução; os sinais são o kmsg e os eventos que o desfazem:
#   1. cada PCI de gpu_unlock.targets= no cmdline tem "gpu_unlock: unlocked" no kmsg deste boot
#      (falta em reboot quente, ordem de carga errada ou placa recusada: só um power-off de verdade resolve);
#   2. o driver nvidia carregou uma vez só e não houve Xid / queda do barramento: reset ou reload volta os
#      seletores de fábrica (o kmsg do boot continua dizendo "unlocked").
# Escreve o estado em /run/gpu-unlock.status (lido pelo MOTD), loga no journal (tag gpu-unlock-check) e sai com
# 1 quando há problema, para a unit ficar failed em `systemctl --failed`.
# GPU_UNLOCK_CMDLINE / GPU_UNLOCK_KMSG / GPU_UNLOCK_STATUS trocam as fontes e o destino; GPU_UNLOCK_NOLOG=1 não loga (teste).
set -uo pipefail
STATUS=${GPU_UNLOCK_STATUS:-/run/gpu-unlock.status}
cmdline=$(cat "${GPU_UNLOCK_CMDLINE:-/proc/cmdline}")
if [ -n "${GPU_UNLOCK_KMSG:-}" ]; then kmsg=$(cat "$GPU_UNLOCK_KMSG"); else kmsg=$(journalctl -k -b --no-pager -o cat); fi
agora=$(date '+%d/%m %H:%M')

estado() {  # nível, mensagem
  echo "gpu-unlock [$agora] $1: $2" > "$STATUS"
  if [ -n "${GPU_UNLOCK_NOLOG:-}" ]; then :
  elif [ "$1" = OK ] || [ "$1" = INFO ]; then logger -t gpu-unlock-check "$1: $2"; else logger -p user.err -t gpu-unlock-check "$1: $2"; fi
  echo "$1: $2"
}

targets=$(grep -oE 'gpu_unlock\.targets=[^ ]+' <<<"$cmdline" | cut -d= -f2)
if [ -z "$targets" ]; then estado INFO "não pedido (sem gpu_unlock.targets no cmdline)"; exit 0; fi
IFS=, read -ra T <<<"$targets"
faltam=()
for t in "${T[@]}"; do
  grep -qiE "$t: gpu_unlock: unlocked" <<<"$kmsg" || faltam+=("$t")
done
cargas=$(grep -c 'NVRM: loading NVIDIA UNIX' <<<"$kmsg")
xids=$(grep -ciE 'NVRM: Xid|fallen off the bus' <<<"$kmsg")

if [ ${#faltam[@]} -gt 0 ]; then
  estado ERRO "$(( ${#T[@]} - ${#faltam[@]} ))/${#T[@]} desbloqueadas; faltam ${faltam[*]}. Desligue de verdade (poweroff, não reboot) e confira de novo"
  exit 1
fi
if [ "$cargas" -gt 1 ] || [ "$xids" -gt 0 ]; then
  estado ALERTA "${#T[@]}/${#T[@]} desbloqueadas no boot, mas depois o driver carregou ${cargas}x e houve ${xids} Xid: o reset pode ter desfeito o unlock. Confira com bench_gemm.py (BF16 ~0,08 TFLOPS/MHz = ok; ~0,042 = travada) ou faça power-off"
  exit 1
fi
estado OK "${#T[@]}/${#T[@]} GPUs desbloqueadas neste boot; driver carregado 1x; sem Xid"
