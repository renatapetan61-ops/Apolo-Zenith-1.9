# Corpus samples: self-authored synthetic text in Brazilian Portuguese,
# English and code snippets. License: authored for this project (no
# third-party rights). Regenerate/extend with scripts/make_corpus.py.
#
# Two outputs:
#   corpus.txt       — original small corpus (23 sentences, 120 repeats)
#   corpus_large.txt — combinatorial corpus (tens of thousands of UNIQUE
#                      sentences from templates; still 100% synthetic and
#                      self-authored — NOT web data)

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


COLORS_PT = ["vermelho", "azul", "verde", "amarelo", "roxo", "laranja",
             "rosa", "branco", "preto", "cinza", "marrom", "dourado"]
COLORS_EN = ["red", "blue", "green", "yellow", "purple", "orange",
             "pink", "white", "black", "gray", "brown", "golden"]
OBJECTS_PT = ["carro", "gato", "cachorro", "casa", "árvore", "montanha",
              "navio", "avião", "livro", "flor", "estrela", "oceano",
              "floresta", "deserto", "ponte", "relógio", "espada", "dragão",
              "cidade", "riacho"]
OBJECTS_EN = ["car", "cat", "dog", "house", "tree", "mountain",
              "ship", "plane", "book", "flower", "star", "ocean",
              "forest", "desert", "bridge", "clock", "sword", "dragon",
              "city", "stream"]
PLACES_PT = ["na estrada", "no jardim", "na cidade", "na praia",
             "na montanha", "no deserto", "na floresta", "no céu",
             "no oceano", "na biblioteca", "no vale", "na ponte"]
PLACES_EN = ["on the road", "in the garden", "in the city", "on the beach",
             "on the mountain", "in the desert", "in the forest", "in the sky",
             "in the ocean", "in the library", "in the valley", "on the bridge"]
TIMES_PT = ["ao amanhecer", "ao pôr do sol", "à noite", "ao meio-dia",
            "na chuva", "na neve", "na neblina", "sob o luar"]
TIMES_EN = ["at dawn", "at sunset", "at night", "at noon",
            "in the rain", "in the snow", "in the fog", "under the moonlight"]


def build_large_corpus() -> str:
    """Combinatorial synthetic corpus: tens of thousands of UNIQUE sentences.

    Templates × colors × objects × places × times (PT + EN) + varied code
    patterns + technical sentences. Still synthetic, still self-authored —
    no web data, no third-party rights.
    """
    out: list[str] = []
    for color in COLORS_PT:
        for obj in OBJECTS_PT:
            for place in PLACES_PT:
                for time_ in TIMES_PT:
                    out.append(f"O {obj} {color} {place} {time_}.")
    for color in COLORS_EN:
        for obj in OBJECTS_EN:
            for place in PLACES_EN:
                for time_ in TIMES_EN:
                    out.append(f"The {color} {obj} {place} {time_}.")
    tech_verbs = ["treina", "avalia", "testa", "documenta", "corrige",
                  "integra", "mede", "registra"]
    tech_objs = ["o backbone", "o tokenizer", "o VAE", "o denoiser",
                 "o roteador", "os experts", "o otimizador", "o checkpoint",
                 "a loss", "os gradientes", "os dados", "os testes"]
    for verb in tech_verbs:
        for tobj in tech_objs:
            for i in range(12):
                out.append(f"O pipeline {verb} {tobj} com passos reais "
                           f"e evidência {i}.")
    fn_names = ["soma", "subtrai", "multiplica", "divide", "potencia",
                "media", "maximo", "minimo", "soma_quad", "norma"]
    ops = ["+", "-", "*", "/"]
    for idx, fn in enumerate(fn_names):
        out.append(f"def {fn}(a, b):")
        out.append(f"    resultado = a {ops[idx % 4]} b")
        out.append("    return resultado")
        out.append(f"class Modulo{idx}:")
        out.append("    def __init__(self, valor):")
        out.append("        self.valor = valor")
        out.append("    def dobra(self):")
        out.append("        return self.valor * 2")
        out.append(f"for i in range({idx * 10}):")
        out.append(f"    print(i, {fn}(i, {idx + 1}))")
        out.append(f"if resultado == {idx}:")
        out.append(f"    print('{fn} ok')")
    return "\n".join(out)


if __name__ == "__main__":
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "sample_corpus" / "corpus.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_corpus(), encoding="utf-8")
    print(f"corpus written: {p} ({p.stat().st_size} bytes)")

    lp = p.parent / "corpus_large.txt"
    large = build_large_corpus()
    lp.write_text(large, encoding="utf-8")
    print(f"corpus_large written: {lp} ({lp.stat().st_size} bytes, "
          f"{len(large.splitlines())} lines, {len(set(large.splitlines()))} unique)")
