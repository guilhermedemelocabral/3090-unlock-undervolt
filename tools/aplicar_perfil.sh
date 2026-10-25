#!/usr/bin/env bash
# aplicar_perfil.sh IDX:PL_W:LOCK_MHZ:OFFSET_MHZ [IDX:PL:LOCK:OFFSET ...] — aplica PL, lock de clock e offset V/F por GPU.
#
# A ordem importa: PL, depois lock (-lgc 210,LOCK), e só então o offset V/F. Offset positivo desloca a curva inteira
# para cima: com o lock ele vira undervolt (o mesmo clock com menos voltagem); SEM o lock, a placa sobe até o topo da
# curva deslocada e o offset vira overclock. Por isso, se o lock falhar, o offset dessa GPU é zerado.
# Nada disso persiste a um reboot ou a um reload do driver: rode no boot (systemd) e, se quiser, num timer.
#
# Exemplos (perfis medidos neste repositório):
#   serving com as 4 GPUs:          sudo ./aplicar_perfil.sh 0:220:1700:120 1:220:1700:120 2:220:1700:120 3:220:1700:120
#   treino com uma GPU, outras ociosas: sudo ./aplicar_perfil.sh 0:450:1950:120
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 1; }
falhas=0
for spec in "$@"; do
  IFS=: read -r i pl lock off <<<"$spec"
  if nvidia-smi -i "$i" -pl "$pl" >/dev/null; then echo "GPU$i: PL ${pl} W"; else echo "GPU$i: ERRO no PL"; falhas=$((falhas + 1)); fi
  if nvidia-smi -i "$i" -lgc "210,$lock" >/dev/null; then
    echo "GPU$i: lock 210-${lock} MHz"
    if python3 -I "$D/vf_offset.py" set "$i=$off" >/dev/null; then echo "GPU$i: offset V/F +${off} MHz"
    else echo "GPU$i: ERRO no offset"; falhas=$((falhas + 1)); fi
  else
    echo "GPU$i: ERRO no lock; zerando o offset (sem lock ele seria overclock)"
    python3 -I "$D/vf_offset.py" set "$i=0" >/dev/null; falhas=$((falhas + 1))
  fi
done
python3 -I "$D/vf_offset.py" get
exit $((falhas > 0))
