# Apolo Zenith 1.9

Framework multimodal de pesquisa (texto/código, imagem, vídeo) com meta
arquitetural de ~199B parâmetros — e com disciplina de evidências: **arquitetura
implementada ≠ capacidade aprendida**. Este repositório contém a base
executável e validada em CPU, o protótipo treinado de verdade e a documentação
do que falta para escalar.

## Início rápido

```bash
pip install -r requirements.txt

# 1) corpus sintético autoral
python3 scripts/make_corpus.py

# 2) testes completos (63 testes)
python3 -m pytest tests/ -q

# 3) treino REAL do protótipo (CPU, ~4s)
python3 scripts/train.py --config configs/prototype_tiny.yaml

# 4) geração de texto com o checkpoint treinado
python3 scripts/generate_text_sample.py

# 5) auditoria dos checkpoints (integridade SHA-256)
python3 scripts/audit_checkpoints.py

# 6) números exatos da meta 199B (sem instanciar)
python3 scripts/plan_199b.py

# 7) pipelines de imagem/vídeo em modo debug (pesos não-treinados: rotulados)
python3 scripts/generate.py --mode video --prompt "um carro vermelho" --debug-generation
# geração FINAL sem pesos treinados é RECUSADA (exit 3) — por design
```

## Documentação

| Documento | Conteúdo |
|---|---|
| `docs/AUDIT_REPORT.md` | Auditoria Fase 1: repositório, hardware, software, causa raiz |
| `docs/ARCHITECTURE.md` | Arquitetura e decisões técnicas |
| `docs/STATUS.md` | Estado honesto por componente (IMPLEMENTED/TESTED/TRAINED/PLANNED/BLOCKED) |
| `docs/SCALING_REQUIREMENTS.md` | Requisitos reais para ~199B (memória, GPUs, dados) |

## Estrutura

```
apolo_zenith/     pacote (config, tokenizer, models, training, generation, inference, utils)
configs/          prototype_tiny (executável), prototype_125m (referência), 199B (meta)
scripts/          train, generate, audit_checkpoints, plan_199b, make_corpus
tests/            63 testes pytest (tokenizador, MoE, backbone, treino real, checkpoints, VAE, vídeo, guards, inventário de linguagens)
data/             corpus sintético autoral (proveniência registrada)
checkpoints/      artefatos de treino (não versionados)
debug/            artefatos do modo debug (não versionados)
```

## Regras do projeto

1. Nenhuma capacidade é declarada sem evidência reproduzível.
2. Geração final com pesos não-treinados é **recusada com erro**, nunca
   produz ruído silencioso.
3. Loading de checkpoint é estrito; relaxações exigem auditoria documentada.
4. Nenhum áudio "nativo" falso; nenhum upscale vendido como resolução nativa.
5. O modelo 199B é meta de design — os pesos NÃO existem.
