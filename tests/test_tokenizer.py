"""Tokenizer tests: losslessness, determinism, persistence (spec §15.9)."""

from pathlib import Path

import pytest

from apolo_zenith.tokenizer import ByteTokenizer, roundtrip_check

TEXTS = [
    "O Apolo Zenith 1.9 audita o repositório antes de mudar o código.",
    "The quick brown fox jumps over the lazy dog.",
    "def soma(a, b):\n    return a + b",
    "acentuação: ção, ãõ, éíóú, ü — emogi 🚀 e símbolos ♥",
    "",
]


@pytest.mark.parametrize("text", TEXTS)
def test_byte_roundtrip_no_merges(text):
    tok = ByteTokenizer()
    assert roundtrip_check(tok, text)


@pytest.mark.parametrize("text", TEXTS)
def test_byte_roundtrip_with_merges(text):
    tok = ByteTokenizer()
    tok.train_merges(" ".join(TEXTS) * 3, num_merges=32)
    assert roundtrip_check(tok, text)
    assert tok.vocab_size == 260 + 32


def test_merges_are_deterministic():
    corpus = "ababab cdcdcd ababab cdcdcd efefef"
    t1 = ByteTokenizer()
    t2 = ByteTokenizer()
    m1 = t1.train_merges(corpus, 10)
    m2 = t2.train_merges(corpus, 10)
    assert m1 == m2


def test_merges_shorten_common_sequences():
    tok = ByteTokenizer()
    corpus = "the theme is therefore the theoretical theorem" * 5
    raw_len = len(ByteTokenizer().encode(corpus))
    tok.train_merges(corpus, 24)
    merged_len = len(tok.encode(corpus))
    assert merged_len < raw_len


def test_special_tokens_encode_decode():
    tok = ByteTokenizer()
    ids = tok.encode("olá", add_bos=True, add_eos=True)
    assert ids[0] == 1 and ids[-1] == 2
    assert tok.decode(ids) == "olá"


def test_save_load_roundtrip(tmp_path: Path):
    tok = ByteTokenizer()
    tok.train_merges("banana bandana banana", 8)
    p = tmp_path / "tokenizer.json"
    tok.save(p)
    tok2 = ByteTokenizer.load(p)
    assert tok2.merges == tok.merges
    assert roundtrip_check(tok2, "banana bandana")


def test_unknown_token_id_raises():
    tok = ByteTokenizer()
    with pytest.raises(ValueError):
        tok.decode([999999])
