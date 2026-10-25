# 3090-unlock-undervolt

Medições de três alavancas de desempenho em placas **GA102 (RTX 3090 / 3090 Ti)**, feitas em outubro de 2026 numa máquina com quatro delas:

1. **Unlock dos tensores** com o [gpu-unlock](https://github.com/Zagitta/gpu-unlock), que tira a meia taxa de BF16/FP16-acc32/TF32 que a NVIDIA impõe às GeForce.
2. **Curva V/F (undervolt):** offset de clock da curva V/F pelo NVML, sempre combinado com um lock de clock.
3. **Power limit (PL):** quanto de potência cada GPU pode puxar.

> **TL;DR (EN):** On RTX 3090/3090 Ti, the tensor unlock doubles BF16 throughput per clock. At a 220 W cap, though, the power limiter eats half of that: +19–26% in dense GEMM and about −6% prefill time for a W4A16 MoE served with vLLM TP4. Decode stays flat.
> For single-GPU training, raising the power limit mattered far more than clocks or offsets. On a 3090 Ti the best measured point was a **1950 MHz lock, +120 MHz V/F offset (undervolt), 450 W**: **~2.3×** the 220 W baseline and **~2.3×** an older run without the unlock.
> The undervolt saves 15–20 W per GPU at the same clock. One step too far shows up as **silent compute errors**, with no crash and no Xid, so validate with bit-exact checks.

## Resumo dos resultados

### Treino com uma GPU (3090 Ti), outras ociosas

O treino foi um fine-tuning QLoRA de 4 bits (rank 8, sequências até 16k) de um modelo denso de ~9B parâmetros, com ~20,1 mil tokens por passo. Medimos pelo tempo real entre passos, sem as pausas de checkpoint. O ruído entre janelas é de ±5–10%, pelo tamanho das amostras.

| 3090 Ti | s/passo | tokens/s | vs. original | vs. treino antigo | potência (mediana / pico) | temp. | anomalias | passos |
|---|---|---|---|---|---|---|---|---|
| *treino antigo: 260 W, ≤1605 MHz, **sem unlock*** | *51,4* | *~683* | — | *1,00×* | — | — | — | *1.258* |
| original: 220 W, 720 MHz, com unlock | 29,15 | ~689 | 1,00× | 1,01× | ~220 W | 58 °C | — | 62 |
| 1800 / +120 / 450 W | 14,54 | ~1.380 | 2,00× | 2,02× | 369 / 425 W | 71 °C | — | 24 |
| 1900 / +60 / 450 W | 11,97 | ~1.680 | 2,44× | 2,46× | 418 / 472 W | 75 °C | — | 35 |
| sem lock / +60 / 450–480 W | 13,48 | ~1.490 | 2,16× | 2,18× | 440–454 / 496 W | 78–80 °C | — | 33 |
| 1900 / +120 / 450 W | 13,37 | ~1.500 | 2,18× | 2,20× | 402 / 475 W | 74 °C | — | 131 |
| 1950 / +120 / 450 W | 12,57 | ~1.600 | 2,32× | 2,34× | 427 / 482 W | 75–77 °C | — | 60 |
| 1950 / +180 / 450 W | 12,27 | ~1.640 | 2,38× | 2,40× | 408 / 485 W | 75 °C | — | 60 |
| 1995 / +180 / 450 W | 11,93 | ~1.680 | 2,44× | 2,47× | ~414 W | 76 °C | — | 28 |
| 2010 / +180 / 450 W (no PL 29% do tempo) | 12,47 | ~1.610 | 2,34× | 2,36× | média 425 W | 78 °C | **2: loss 6×10³¹, grad_norm inf** | 60 |
| **1950 / +120 / 450 W (sweet spot)** | **12,84** | **~1.560** | **2,27×** | **2,29×** | 426 / 487 W | 73–76 °C | — | **178** |

- O formato da coluna de configuração é *lock de clock / offset V/F / PL*.
- O treino antigo usou outro método (LoRA BF16, seq 8k, ~35,1 mil tokens por passo), então a comparação com ele é aproximada.
- **Liberar a potência deu o salto**, de 220 para 450 W: 2×. De 1900 a 2010 MHz tudo fica na mesma faixa (12–13,4 s/passo), dentro do ruído.
- **O undervolt aparece na potência:** com o mesmo clock, +60 → +120 economizou ~15 W e +120 → +180 economizou ~19 W.
- **2010 / +180 errou conta em silêncio.** Houve dois registros anômalos em ~215 passos: um loss de 6×10³¹ com o grad_norm normal (típico de um logit corrompido, porque o gradiente do softmax é limitado) e um grad_norm infinito. Nenhum apareceu antes nem depois.
  - O offset validado no stress com ~180 W de carga não vale com ~425 W. A corrente maior consome a margem de voltagem.
- **Sem lock é pior:** a 78–80 °C o GPU Boost corta clock, e os watts a mais viram calor.

### Unlock dos tensores (GEMM, 220 W)

| GPU | BF16 TFLOPS/MHz stock → unlock | BF16 TFLOPS líquidos stock → unlock |
|---|---|---|
| RTX 3090 Ti | 0,043 → 0,084 | 38,8 → 46,2 (**+19%**) |
| RTX 3090 | 0,042 → 0,081 | 44,7 → 55,0 (**+23%**) |
| RTX 3090 | 0,042 → 0,081 | 49,9 → 62,8 (**+26%**) |
| RTX 3090 | 0,042 → 0,081 | 44,0 → 55,1 (**+25%**) |

- **Controle:** FP16 acumulando em FP16, que já tem taxa cheia, ficou em ~0,08 nas duas condições.
- **Por que o ganho líquido é menor:** o dobro por clock não vira o dobro de TFLOPS porque, a 220 W, o limitador derruba o clock de ~1000 para ~650 MHz.
- **Soak conjunto de 10 min** com as quatro desbloqueadas: sem Xid, p99 de 220 W, 77 °C.
- Detalhes em [`results/gemm_unlock_220w.csv`](results/gemm_unlock_220w.csv) e [`results/soak_conjunto_unlock_220w.csv`](results/soak_conjunto_unlock_220w.csv).

### Inferência (vLLM, MoE W4A16, TP4, 4 GPUs a 220 W)

Modelo MoE com ~6B parâmetros ativos (família Qwen3.8-Flash-Next, GPTQ W4A16 / Marlin), vLLM TP4 + DCP2, MTP k=3, KV FP8. Uma bateria completa por braço.

| métrica | stock 1600 MHz | unlock 1600 MHz | unlock 1700 / +120 |
|---|---|---|---|
| prefill frio de 84k (n=12, CV ≤ 0,2%) | 52,5 s | 49,5 s (−5,7%) | 48,8 s |
| prefill frio de ~150k | 98,1 s | 91,8 s (−6,4%) | 93,5 s |
| decode prosa / código (1 fluxo) | 126 / 186 tok/s | 128 / 188 | 130 / 192 |
| 6 fluxos agregados | 370 tok/s | 388 | 393 |

- O decode praticamente não muda, porque é limitado pela banda de memória.
- O serving não encosta em 220 W: fica preso no lock de clock.
- Num ABA na mesma sessão, 1700/+120 contra 1600/0 deu **+2,6–3,8% de decode**, com até ~8 W a mais por GPU.
- Em [`results/inferencia_moe_w4a16_tp4.csv`](results/inferencia_moe_w4a16_tp4.csv).

### Fronteira do undervolt (stress bit a bit, 4 GPUs juntas, 220 W, 180 s por degrau)

| lock | +60 | +120 | +180 | +240 |
|---|---|---|---|---|
| 1600 MHz | ok | ok | ok | ok |
| 1700 MHz | ok | ok | ok | **uma 3090 divergiu bit a bit em 16 s**, sem Xid |
| 1800 MHz | só a 3090 Ti passou de ~1725 MHz; ok até +180 | | | |

- **Com o mesmo clock e a mesma potência** (~180 W), +240 deixou as 3090 fazerem **18–22% mais trabalho**. A 3090 Ti satura em +120 a 1600 MHz: já está no piso de voltagem.
- **Teto das 3090:** com lock de 1800, elas param em ~1725 MHz sem nenhum motivo de throttle reportado pelo NVML. Em TP com várias GPUs, um lock acima disso não ajuda.
- **Overclock de memória** (até o clock do estado P0 e +5%) não deu banda nenhuma sob o PL, só potência.
- Em [`results/escada_undervolt_220w.csv`](results/escada_undervolt_220w.csv).

### A fonte (1250 W): o PL é média, não teto

| carga | por GPU, mediana / pico de 100 ms | GPUs + CPU (máx.) | estimativa com +60 W do resto |
|---|---|---|---|
| stress de memória, 4 GPUs a 220 W | 214–229 / 234–265 W | 1.082 W | ~1.140 W (**~91%**) |
| inferência MoE TP4 (1700/+120) | 167–198 / 180–210 W | 885 W | ~945 W (~76%) |
| treino numa GPU (1950/+120, 450 W) | 426 / 487 W | ~620 W | ~680 W (~54%) |

- O +12 V (Super I/O da placa-mãe, 1 Hz) não passou de 11,90 V, mas essa leitura não vê transientes de milissegundos.
- A regra que seguimos: **limitador solto só com uma GPU em carga**.
- Em [`results/fonte_1250w.csv`](results/fonte_1250w.csv).

## Máquina

| | |
|---|---|
| GPUs | 1× RTX 3090 Ti (EVGA) + 3× RTX 3090, GA102, 24 GB cada |
| PCIe | 4.0, sem NVLink e sem P2P: x4 (3090 Ti, lanes da CPU), x8 atrás do chipset, x8 + x8 (CPU) |
| CPU / placa | Ryzen 7 5800X3D / ASUS Pro WS X570-ACE |
| Fonte | 1250 W |
| SO / driver | Ubuntu 24.04, kernel 6.8, NVIDIA 595.99.02 (open) |
| Unlock | [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) em `af28cbb` + `dbe10a5` (loader do initramfs do Ubuntu), `gpu_unlock.resume=0` |

## Método

- **Undervolt pelo NVML documentado:** `nvmlDeviceSetGpcClkVfOffset` (offset em MHz, múltiplo de 15) **depois** de `nvidia-smi -lgc 210,LOCK`. Com o lock, a GPU roda LOCK MHz com a voltagem que a curva de fábrica usa em LOCK − offset. Sem o lock, o mesmo offset é overclock.
  - A API de memória (`nvmlDeviceSetMemClkVfOffset`) usa **2× o MHz** que o `nvidia-smi` mostra: +500 deu +250 MHz.
- **Estabilidade por conferência bit a bit:**
  - `tools/stress_vf.py` segura a GPU no clock do lock, perto do PL, e compara cada GEMM com o primeiro resultado.
  - `tools/stress_mem.py` faz o mesmo com cópias de memória.
  - Erro de undervolt aparece como **conta errada sem crash**. "Não travou" não prova nada.
- **Velocidade de treino:** pelo tempo real entre passos do tqdm, sem as pausas de checkpoint. Configurações diferentes caem em janelas com amostras diferentes, daí o ruído de ±5–10%.
- **Contador, não bit:** o bit instantâneo de "HW Slowdown" aparecia ativo numa 3090 Ti, mas os contadores acumulados do driver marcaram 0 µs de freio térmico e de potência. Use os contadores (`nvidia-smi -q -d PERFORMANCE`).

## Ferramentas (`tools/`)

| arquivo | o que faz |
|---|---|
| `bench_gemm.py` | GEMM 8192³ por tipo (BF16/FP16/TF32), em TFLOPS por MHz de SM; `--soak N` para carga contínua |
| `vf_offset.py` | lê e aplica os offsets V/F de GPC e de memória por GPU (NVML, root) |
| `aplicar_perfil.sh` | aplica PL, lock e offset por GPU, nessa ordem; zera o offset se o lock falhar |
| `stress_vf.py` | stress no clock do lock, perto do PL, com conferência bit a bit |
| `degrau_vf.sh` | um degrau da escada de undervolt em todas as GPUs, com watchdog: Xid, temperatura, PL, freio de hardware e divergência |
| `stress_mem.py` | banda e integridade da GDDR6X, para testar offset de memória |
| `monitor_consumo.py` | consumo do sistema por segundo: GPUs (NVML, instantâneo), CPU (RAPL), carga da fonte e trilhos +12/+5/+3,3 V pelo Super I/O (mapeamento NCT6798D da ASUS; confira o seu) |
| `gpu-unlock-check.sh` | confere, a cada boot e periodicamente, se cada GPU do `gpu_unlock.targets` desbloqueou e se o driver não recarregou nem teve Xid (o unlock some num reset) |

Exemplos:

```bash
# base de TFLOPS/MHz de uma GPU (≈0,042 travada, ≈0,08 destravada)
docker run --rm --gpus '"device=0"' -v "$PWD/tools":/bench:ro --entrypoint python3 \
  vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c /bench/bench_gemm.py

# um degrau da escada: lock 1700 MHz, +120 em 4 GPUs, 180 s, com o PL que estiver aplicado (PL=220 por padrão)
sudo -v && PL=220 ./tools/degrau_vf.sh L1700-o120 1700 180 120 120 120 120

# perfil de treino numa GPU só (as outras ociosas)
sudo ./tools/aplicar_perfil.sh 0:450:1950:120
```

## Riscos

- **O unlock** carrega firmware assinado e reescreve seletores da GPU antes do driver. Leia o README do gpu-unlock. Ele some com um reset ou reload do driver, e um reboot quente pode ser recusado: desligue de verdade.
- **O undervolt** pode errar conta **sem travar**. Valide com conferência bit a bit e no seu workload real (loss, grad_norm, saídas). Deixe margem: aqui, +180 passou a 1800 MHz com ~180 W e falhou a 2010 MHz com ~425 W.
- **A fonte:** o PL é média. Picos de 100 ms passam do PL, e quatro GPUs em carga de memória levaram uma fonte de 1250 W a ~90% (estimado).
- **Atualizações de driver ou kernel** podem quebrar o unlock. Aqui elas foram retiradas do unattended-upgrades e postas em `apt-mark hold`.

Use por sua conta e risco.

## Créditos

- **[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock)**, autor do módulo que destrava a taxa dos tensores nas GA10x. Os testes de unlock aqui só foram possíveis graças a ele. Os relatos de compatibilidade nas issues do projeto (por exemplo #15, #25 e #75) guiaram as expectativas.
- **[noonghunna/club-3090](https://github.com/noonghunna/club-3090):** medições comunitárias em 3090 (W4A8 contra W4A16, P2P).
- **[NVIDIA NVML](https://docs.nvidia.com/deploy/nvml-api/):** API de offsets V/F, locks e contadores de throttle.
- **[Henry Freed, "GPU Power Limits and Undervolting on Linux"](https://hbfreed.com/2026/03/12/power-limits-undervolting.html):** lock + offset + PL em 3× 3090 no Linux.
- **Fóruns de undervolt de Ampere** (Igor's Lab, ComputerBase, TechPowerUp, Overclockers UK, EVGA): pontos de referência de V/F das 3090/3090 Ti.
- **[vLLM](https://github.com/vllm-project/vllm):** motor de inferência usado nos testes.

## Licença

Os scripts e textos deste repositório estão sob a licença MIT (ver `LICENSE`). O gpu-unlock tem a própria licença e não é redistribuído aqui.
