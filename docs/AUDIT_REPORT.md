# Relatório de Auditoria — Apolo Zenith 1.9 (Fase 1)

Data: 2026-10-10 · Auditor: Cline (sessão de desenvolvimento)

## 1. Estado do repositório (evidência: `git log`, `git status`, `find`)

| Item | Estado |
|---|---|
| Commits | 1 — `d7aed84 "Add files via upload"` |
| Branch | `main` (limpo, sem alterações pendentes) |
| Arquivos antes do trabalho | Apenas `Apolo_Zenith_1.9_Prompt_Mestre_Cline.md` |
| Código pré-existente | **NENHUM** |
| Checkpoints pré-existentes | **NENHUM** |
| Datasets pré-existentes | **NENHUM** |
| Testes pré-existentes | **NENHUM** |

**Causa raiz do "vídeo com ruído 128×128" citado no prompt mestre (§15):**
não há pipeline, pesos ou artefatos neste repositório que possam ser
auditados. O vídeo ruidoso relatado não está acessível aqui (nenhum
caminho foi fornecido). Conforme o prompt determina, nenhuma análise
específica daquele arquivo foi inventada; em vez disso, o pipeline de vídeo
foi (re)construído com verificações explícitas de layout `[B,C,T,H,W]`,
asserts de finitude e guards anti-pesos-aleatórios (ver
`apolo_zenith/generation/` e `tests/test_guards.py`).

## 2. Hardware (evidência: comandos executados no ambiente)

| Recurso | Valor | Comando/evidência |
|---|---|---|
| CPU | 4 vCPU | `nproc` |
| RAM | 15 GiB | `free -h` |
| Disco livre | ~19 GB | `df -h /workspaces` |
| GPU | **NENHUMA** | `nvidia-smi` bloqueia indefinidamente (sem driver); `torch.cuda.is_available() == False` |
| CUDA/NVCC | ausente | sem toolkit |

Consequências:
- Treinar 199B: **IMPOSSÍVEL aqui** (estimativas reais em
  `docs/SCALING_REQUIREMENTS.md`).
- Treinar/validar protótipos pequenos em CPU: **POSSÍVEL e EXECUTADO**
  (ver `checkpoints/tiny/training_log.jsonl` após rodar `scripts/train.py`).

## 3. Software (evidência: `pip list`, imports verificados)

| Pacote | Versão instalada |
|---|---|
| Python | 3.14.2 |
| torch | 2.14.1+cpu (sem CUDA) |
| numpy | 2.5.3 |
| pytest | 9.1.1 |
| Pillow | 12.3.0 |
| safetensors | instalado |
| imageio / imageio-ffmpeg | 2.38.1 + ffmpeg v7.0.2 embutido |
| PyYAML | presente (pré-instalado) |
| ffmpeg do sistema | ausente (usamos o binário embutido do imageio-ffmpeg) |

Nota de engenharia: `nvidia-smi` neste Codespace trava o terminal em estado
ininterrupto — os scripts de auditoria **não** o invocam; a presença de GPU
é detectada apenas via `torch.cuda.is_available()`.

## 4. O que foi implementado a partir da auditoria

Estrutura criada (ver `docs/STATUS.md` para o estado exato de cada módulo):
- `apolo_zenith/` — pacote: config, tokenizer byte-level, backbone
  Transformer+MoE, VAE conv, transformador de vídeo latente, roteador de
  modalidades, trainer com logging real, checkpoints com SHA-256, guards de
  geração, modo debug.
- `scripts/` — treino, geração, auditoria de checkpoints, plano 199B.
- `tests/` — suíte pytest (executada; ver seção 5).
- `configs/` — protótipo tiny (executável aqui), 125M (referência), 199B
  (meta PLANNED, nunca instanciada).
- `data/sample_corpus/` — corpus sintético autoral (sem direitos de
  terceiros), gerado por `scripts/make_corpus.py`.

## 5. Reprodução de falhas (antes da implementação)

Não havia código para reproduzir falhas. As verificações equivalentes
foram definidas como testes que falham sem implementação correta:
integridade de checkpoint (adulteração detectada), loading estrito
(chaves ausentes levantam erro), layouts de vídeo (reshape cego é proibido),
guards (pesos não-treinados bloqueiam geração final).

## 6. Plano priorizado executado

1. ✅ Auditoria de repositório/hardware/dependências (este documento).
2. ✅ Fundação: config + tokenizer + backbone MoE + VAE + vídeo latente.
3. ✅ Treino real (escala protótipo) com evidências registradas em log.
4. ✅ Checkpoints: integridade SHA-256, strict loading, resume de RNG.
5. ✅ Guards de geração + modo debug com artefatos honestos.
6. ✅ Testes pytest executados (resultados em `docs/STATUS.md`).
7. ✅ Requisitos de escala documentados (`docs/SCALING_REQUIREMENTS.md`).
