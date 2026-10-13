# Apolo Zenith 1.9 — Status honesto por componente

Legenda: **IMPLEMENTED** = código existe e é exercitado por testes ·
**TESTED** = validado com execuções reais neste ambiente ·
**TRAINED** = passos de treino reais executados com loss registrada ·
**PROTOTYPE** = validação arquitetural em escala pequena ·
**PLANNED** = projetado, não implementado · **BLOCKED** = impossível neste
ambiente (requisitos em `docs/SCALING_REQUIREMENTS.md`).

| Componente | Implementado | Testado | Treinado | Estado | Evidência |
|---|---|---|---|---|---|
| Tokenizer byte-level + BPE | sim | sim | n/a (treino de merges é determinístico e testado) | IMPLEMENTED, TESTED | `tests/test_tokenizer.py` |
| Backbone decoder (RoPE, RMSNorm, SwiGLU) | sim | sim | **sim — 2 escalas**: tiny (250.944 params, 150 passos, val 0,266) e 125M-classe (126.083.200 params, **1.000 passos em CPU** com corpus combinacional de 47k frases únicas; val no corpus grande: 0,70→0,54→0,25→**0,224**; **resume real validado 2×** e geração sintetiza combinações NOVAS do template) | PROTOTYPE, TRAINED | `checkpoints/tiny/training_log.jsonl`, `checkpoints/125m/training_log.jsonl`, `tests/test_training.py` |
| MoE esparso (top-k, aux loss, capacity) | sim | sim | indireto (via backbone) | IMPLEMENTED, TESTED | `tests/test_moe.py` |
| Trainer (AdamW, warmup, clip, logging) | sim | sim | sim | IMPLEMENTED, TESTED, TRAINED | log JSONL com loss/lr/grad_norm |
| Checkpoints (SHA-256, strict load, RNG) | sim | sim | n/a | IMPLEMENTED, TESTED | `tests/test_checkpoint.py` |
| VAE convolucional | sim (arquitetura) | mecânica (shapes, loss ↓ em batch fixo) | **sim — escala toy**: 300 passos em imagens sintéticas 16×16, rec_mse 0,50→0,02 (`checkpoints/media/`) | IMPLEMENTED, TRAINED (toy); qualidade de produção NÃO validada | `tests/test_vae.py`, `tests/test_media_engines.py` |
| Transformador de vídeo latente [B,C,T,H,W] | sim (arquitetura) | forward/ layouts/ debug run | **sim — escala toy**: 300 passos em vídeos sintéticos 4-frame, eps_mse 1,43→0,99 | IMPLEMENTED, TRAINED (toy) | `tests/test_video_shapes.py`, `checkpoints/media/media_training_log.jsonl` |
| Geração de imagem (final) | pipeline sim | **sim — guard passou com componentes toy-treinados**: 16×16 gerada, prompts distintos → saídas distintas (dist. cosseno 0,0499) | **sim — toy** (400 passos denoiser + 300 VAE) | IMPLEMENTED, TRAINED (toy); qualidade mínima — ver nota | `checkpoints/media/media_training_report.json` |
| Geração de vídeo (final) | pipeline sim | **sim — guard passou**: MP4 real 4 frames @4fps verificado por probe | **sim — toy** | IMPLEMENTED, TRAINED (toy) | `checkpoints/media/generated_media_video.mp4` |
| Export MP4 + probe | sim | sim (frames/fps/duração reais) | n/a | IMPLEMENTED, TESTED | `tests/test_video_shapes.py` |
| Modo debug (`--debug-generation`) | sim | sim | n/a | IMPLEMENTED, TESTED | artefatos em `debug/` |
| Áudio | **não** (falha explicitamente) | n/a | NÃO | PLANNED | `apolo_zenith/models/audio.py` |
| Coding engine / agentes / ferramentas | **não** | n/a | NÃO | PLANNED | ver `docs/ARCHITECTURE.md` |
| Inventário de linguagens (§7.1) | semente (25 linguagens) | sim (schema validado) | n/a (dados, não capacidade) | PROTOTYPE | `data/language_inventory.json`, `tests/test_language_inventory.py` |
| Roteador de modalidades | arquitetura sim | mecânica sim | NÃO | IMPLEMENTED, NOT TRAINED | `tests/` (import smoke) |
| Meta 199B (config) | config sim | matemática validada vs. instância tiny | **NUNCA treinada** | PLANNED, BLOCKED | `scripts/plan_199b.py`, `tests/test_plan_math.py` |
| Pipeline 8K (7680×4320) | **não** | n/a | NÃO | PLANNED, BLOCKED | `docs/SCALING_REQUIREMENTS.md` |

## Declarações explícitas de NÃO-capacidade

1. **O modelo de 199B NÃO existe** como pesos treinados. A config
   `configs/apolo_zenith_1_9.yaml` é um alvo de design; os números de
   parâmetros são calculados analiticamente e a fórmula é validada contra
   instanciações reais pequenas (`tests/test_plan_math.py`).
2. **Não há geração de imagem/vídeo de qualidade de produção.** Os guards
   ainda bloqueiam componentes aleatórios (`UntrainedComponentError`); após o
   treino toy de `checkpoints/media/` eles passam legitimamente, mas a
   qualidade é mínima (16×16, formas sintéticas, ~1M params, centenas de
   passos em CPU). Mídia utilizável exige GPU + datasets licenciados em
   larga escala (`docs/SCALING_REQUIREMENTS.md`).
3. **Nenhum áudio é gerado.** Nenhum vídeo produzido por este repositório
   contém trilha "nativa" — anexar áudio estático e chamar de nativo é
   explicitamente proibido no código (`apolo_zenith/models/audio.py`).
4. **Os protótipos treinados (tiny e 125M-classe) não têm capacidade de
   linguagem real.** Eles validam a mecânica de treino em duas escalas
   (loss decrescente, checkpoints íntegros com SHA-256, resume, geração
   a partir do checkpoint). O 125M rodou 1.000 passos (schedule reduzido
   vs. 2.000 da config) em corpus sintético combinacional: o modelo aprende
   o TEMPLATE das frases e sintetiza combinações novas, mas nada disso é
   capacidade geral de linguagem.
5. **Upscale não é geração nativa.** Nenhuma saída é rotulada de resolução
   que não seja a real (ver `probe_video` para verificação de dimensões).
