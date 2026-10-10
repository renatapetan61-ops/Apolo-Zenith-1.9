# Arquitetura — Apolo Zenith 1.9

## Visão geral

```
Text/Code ─┐
           │  ByteTokenizer (256 bytes + merges determinísticos)
Visão ─────┤  ConvVAE (encoder de latentes, protótipo: 2D espacial)
Vídeo ─────┤  LatentVideoTransformer sobre [B,C,T,H,W]
Áudio ─────┘  (PLANNED — falha explicitamente, sem placeholders falsos)
           ↓
   Multimodal Conditioning (TextCondEncoder / futuras fusões)
           ↓
   Shared Reasoning Backbone (decoder-only, RoPE, RMSNorm, SwiGLU)
           ↓
   Modality/Task Router (ModalityRouter, aprendido)
     ↙        ↓        ↘
  Coding   Imagem     Vídeo/Áudio (heads generativos sobre latentes)
```

## Componentes implementados (todos com testes reais)

### Backbone (`apolo_zenith/models/backbone.py`)
- Decoder-only; blocos pre-norm RMSNorm; atenção causal multi-cabeça com
  RoPE (par de canais rotacionado); FFN SwiGLU denso **ou** MoE esparso.
- Embeddings atados (tied) ao head de saída.
- `generate()` com temperatura/top-k para inferência.

### MoE (`apolo_zenith/models/moe.py`)
- Top-k routing por token; pesos de dispatch normalizados.
- Aux loss de balanceamento (Switch-Transformer); capacity factor com
  **drop explícito** (nunca silencioso; contado em `last_dropped_tokens`).
- Cada expert é SwiGLU(h→i→h) sem bias.

### VAE (`apolo_zenith/models/vision.py`)
- Convolucional, estilo Stable Diffusion: downsample apenas entre estágios
  (fator 2^(len(mult)-1)); cabeças μ/logσ²; decoder espelhado.
- Estado: **arquitetura implementada; NÃO treinada para produção** (a validação
  de mecânica de treino existe e é honesta sobre seu alcance).

### Vídeo latente (`apolo_zenith/models/video.py`)
- Transformer sobre latentes **[B,C,T,H,W]** com atenção espacial por frame +
  atenção temporal entre frames + FFN — frames trocam informação de verdade
  (codificar frames independentes é proibido por design, spec §10).
- Objetivo único: predição de epsilon (DDPM). Nenhuma mistura de
  v-prediction/DDIM/flow-matching no mesmo pipeline (spec §14.5).
- Todos os reshapes são validados com asserts de layout.

### Pipelines de geração (`apolo_zenith/generation/`)
- Imagem = vídeo com T=1: um único denoiser latente serve aos dois.
- DDPM com schedule linear; sampler único e consistente.
- **Guards**: geração final exige, por componente, checkpoint com
  SHA-256 íntegro + contagem de parâmetros conferindo + `trained: true`
  com passos > 0. Sem isso: `UntrainedComponentError` (exit 3 na CLI).
- **Debug mode** (`--debug-generation`): executa o pipeline com pesos
  não-treinados e grava artefatos rotulados "UNTRAINED — NOT A VALID
  GENERATION": model_load_report, text_embedding_stats, latent_stats,
  passos de denoise, frame final, MP4 e generation_report.json.

### Treino (`apolo_zenith/training/`)
- AdamW (β=0.9/0.95), warmup linear + cosseno até 10% do pico,
  clip de gradiente, aux loss MoE ponderada.
- Log JSONL por evento (step, loss, lr, grad_norm, tokens, segundos).
- Checkpoints `latest` e `best` **sem se sobrescreverem**; metadados com
  SHA-256 dos pesos, RNG do PyTorch e flags `trained/steps/final_loss`.
- `resume()` restaura pesos, otimizador, contador de passos e RNG (testado
  com modelo perturbado para detectar falhas de restauração).

### Segurança de carregamento (`apolo_zenith/training/checkpoint.py`)
- Loading é **estrito por padrão**: chaves ausentes/inesperadas/shape-mismatch
  levantam erro unificado com auditoria completa.
- `strict=False` nunca oculta nada: retorna relatório com TODAS as listas
  (missing/unexpected/shape_mismatch).
- Integridade SHA-256 verificada antes de qualquer load; adulteração detectada
  bloqueia a geração (testado).

## Escala

- Protótipo executado aqui: 250.944 parâmetros (treinado de verdade, ver
  `checkpoints/tiny/training_log.jsonl`).
- Meta ~199B: ver `docs/SCALING_REQUIREMENTS.md` (200,19B total / 28,7B
  ativos, fórmula validada; STATUS: PLANNED, nunca instanciada).
