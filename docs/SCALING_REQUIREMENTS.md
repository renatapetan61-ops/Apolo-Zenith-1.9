# Requisitos reais para escalar o Apolo Zenith 1.9 até ~199B

Este documento contém **números calculados e verificáveis**, gerados por
`scripts/plan_199b.py` (a fórmula analítica é validada contra uma instância
real: 250.944 parâmetros previstos = 250.944 contados). Nada aqui é estimativa
de marketing.

## 1. Configuração-alvo (configs/apolo_zenith_1_9.yaml — STATUS: PLANNED)

| Parâmetro | Valor |
|---|---|
| hidden_size | 7.168 |
| num_layers | 59 |
| experts por camada | 96 (top-8 por token) |
| FFN por expert | 1.536 (SwiGLU, sem bias) |
| vocabulário | 131.072 (byte-level + merges) |
| max_seq_len (meta) | 32.768 (extensão a 131k: PLANNED) |

## 2. Contagem de parâmetros (calculada, não treinada)

| Métrica | Valor |
|---|---|
| **Total** | **200.189.445.120 (~200,19B)** — meta do spec ~199B, diferença +0,6% |
| **Ativos por token** | **28.696.937.472 (~28,7B)** (top-8 de 96 experts) |
| Embedding (tied) | 939.524.096 |
| Atenção por camada | 205.520.896 (MHA 4h²; GQA reduziria ~1/3 — PLANNED) |
| Router por camada | 688.128 |
| Experts por camada (96×SwiGLU) | 3.170.893.824 |

Validação: a mesma fórmula aplicada ao protótipo tiny dá **exatamente**
250.944 parâmetros, conferindo com a instância real (`tests/test_plan_math.py`).

## 3. Memória de treinamento (AdamW + bf16 + master fp32)

| Estado | Bytes/parâm | Total |
|---|---|---|
| Pesos bf16 | 2 | 400,4 GB |
| Gradientes bf16 | 2 | 400,4 GB |
| Adam m (fp32) | 4 | 800,8 GB |
| Adam v (fp32) | 4 | 800,8 GB |
| Pesos master fp32 | 4 | 800,8 GB |
| **Total estático** | 16 | **3,20 TB** |

Ativações com gradiente-checkpointing completo: ~55 GB por micro-batch
(8×8.192 tokens × 59 camadas × 7.168 hidden × 2 bytes). Sem checkpointing
seria ~100× maior — inviável.

## 4. Hardware mínimo realista

| Cenário | GPUs H100 80GB |
|---|---|
| Só estados estáticos (FSDP/ZeRO-3 completo) | ~44 |
| **Piso prático** (ativas + EP/DP/PP + redundância) | **~128** |
| Confortável (throughput, falhas, eval) | 256+ |

Comparação de referência pública: modelos MoE abertos de escala semelhante
(ex.: DeepSeek-V3, 671B totais / 37B ativos) foram treinados em clusters de
milhares de GPUs por meses. 199B está abaixo disso, mas a ordem de grandeza
de cluster se mantém: **centenas de H100/H200/MI300X por semanas a meses**.

## 5. Dados (requisitos, sem promessas)

- Texto/código: trilhões de tokens com licenças registradas por fonte
  (pipeline Collect→Licença→Filtro→Dedup→Shard, spec §14.4).
- Imagem/vídeo: datasets com licença explícita de treino (ex.: LAION-variantes
  permissivas, datasets sintéticos), com proveniência e checksum por shard.
- Nenhum dataset foi usado além do corpus sintético autoral deste repositório.

## 6. O que este ambiente (4 vCPU, 15 GiB RAM, sem GPU) NÃO pode fazer

- Instanciar os 200B (só os estados estáticos seriam 3,2 TB).
- Treinar qualquer componente multimodal com qualidade utilizável.
- Gerar vídeo 8K (7680×4320): exige VAE de vídeo + super-resolução treinados
  e GPU com dezenas de GB de VRAM apenas para o decode.

## 7. Caminho de escala (ordem recomendada)

1. 125M em 1×A100/H100 (config `prototype_125m.yaml`): valida GQA, Flash
   Attention, contexto 1k, checkpointing.
2. 1B–3B em 1 nó (8×GPU): valida MoE em escala real, balanceamento de carga,
   expert parallelism.
3. 30B–70B em multi-nó: valida FSDP+EP+PP, retomada distribuída, bf16/fp8.
4. 199B: apenas quando 1–3 estiverem verificados com métricas publicadas.
