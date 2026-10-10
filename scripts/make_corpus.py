# Corpus samples: self-authored synthetic text in Brazilian Portuguese,
# English and code snippets. License: authored for this project (no third-party
# rights). Regenerate/extend with scripts/make_corpus.py.

def samples() -> list[str]:
    pt = [
        "O Apolo Zenith 1.9 é um projeto de modelo multimodal.",
        "O engenheiro audita o repositório antes de mudar o código.",
        "Cada teste executado registra evidência real no log de treino.",
        "Um carro esportivo vermelho numa estrada ao pôr do sol.",
        "Um gato branco dormindo num sofá azul na sala.",
        "O modelo aprende padrões de linguagem a partir de dados.",
        "O rio atravessa a montanha em direção ao vale verde.",
        "A luz da manhã entra pela janela da biblioteca.",
    ]
    en = [
        "The quick brown fox jumps over the lazy dog.",
        "A neural network learns functions from examples.",
        "Sparse mixture of experts activates a subset of parameters per token.",
        "Backpropagation computes gradients of the loss with respect to weights.",
        "The repository contains tests, configs and training scripts.",
    ]
    code = [
        "def soma(a, b):",
        "    return a + b",
        "class Ponto:",
        "    def __init__(self, x, y):",
        "        self.x = x",
        "        self.y = y",
        "for i in range(10):",
        "    print(i * i)",
        "if nome == 'apolo':",
        "    print('zenith')",
    ]
    return pt + en + code


def build_corpus(repeat: int = 120) -> str:
    lines = samples()
    out = []
    for i in range(repeat):
        out.extend(lines)
    return "\n".join(out)


if __name__ == "__main__":
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "sample_corpus" / "corpus.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_corpus(), encoding="utf-8")
    print(f"corpus written: {p} ({p.stat().st_size} bytes)")
