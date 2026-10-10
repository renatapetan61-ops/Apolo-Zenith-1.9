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
| Backbone decoder (RoPE, RMSNorm, SwiGLU) | sim | sim | **sim (protótipo tiny)** | PROTOTYPE, TRAINED | `checkpoints/tiny/training_log.jsonl`, `tests/test_training.py` |
| MoE esparso (top-k, aux loss, capacity) | sim | sim | indireto (via backbone) | IMPLEMENTED, TESTED | `tests/test_moe.py` |
| Trainer (AdamW, warmup, clip, logging) | sim | sim | sim | IMPLEMENTED, TESTED, TRAINED | log JSONL com loss/lr/grad_norm |
| Checkpoints (SHA-256, strict load, RNG) | sim | sim | n/a | IMPLEMENTED, TESTED | `tests/test_checkpoint.py` |
| VAE convolucional | sim (arquitetura) | mecânica (shapes, loss ↓ em batch fixo) | **apenas mecânica** — qualidade NÃO validada | IMPLEMENTED, NOT PRODUCTION-TRAINED | `tests/test_vae.py` |
| Transformador de vídeo latente [B,C,T,H,W] | sim (arquitetura) | forward/ layouts/ debug run | **NÃO** | IMPLEMENTED, NOT TRAINED | `tests/test_video_shapes.py` |
| Geração de imagem (final) | pipeline sim | **bloqueada por guard até existir pesos treinados** | NÃO | IMPLEMENTED, BLOCKED (dados/GPU) | `tests/test_guards.py` |
| Geração de vídeo (final) | pipeline sim | **bloqueada por guard** | NÃO | IMPLEMENTED, BLOCKED | `tests/test_guards.py` |
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
2. **Não há geração de imagem/vídeo "pronta"**: os pipelines recusam-se a
   produzir saída final com pesos não treinados (`UntrainedComponentError`).
   O modo debug executa o pipeline, mas marca todos os artefatos como
   "UNTRAINED — NOT A VALID GENERATION".
3. **Nenhum áudio é gerado.** Nenhum vídeo produzido por este repositório
   contém trilha "nativa" — anexar áudio estático e chamar de nativo é
   explicitamente proibido no código (`apolo_zenith/models/audio.py`).
4. **O protótipo tiny treinado não tem capacidade de linguagem real.** Ele
   valida a mecânica de treino (loss decrescente, checkpoints, resume);
   memorizar 23 frases sintéticas não é evidência de capacidade geral.
5. **Upscale não é geração nativa.** Nenhuma saída é rotulada de resolução
   que não seja a real (ver `probe_video` para verificação de dimensões).
