"""Byte-level tokenizer for Apolo Zenith 1.9 (IMPLEMENTED, TESTED).

Every byte 0-255 is a token (guaranteed lossless round-trip), plus special
tokens and an optional deterministic byte-pair merge table trained from a
corpus. Dependency-free so the prototype runs anywhere.

Vocab layout:
  ids 0-3    : special tokens (<|pad|>, <|bos|>, <|eos|>, <|unk|>)
  ids 4-259  : the 256 raw bytes
  ids 260+   : learned merges (each merge = (left_id, right_id))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

SPECIAL_TOKENS: Dict[str, int] = {
    "<|pad|>": 0,
    "<|bos|>": 1,
    "<|eos|>": 2,
    "<|unk|>": 3,
}
SPECIAL_IDS = set(SPECIAL_TOKENS.values())
BYTE_ID_OFFSET = 4
NUM_BYTES = 256
FIRST_MERGE_ID = BYTE_ID_OFFSET + NUM_BYTES  # 260


class ByteTokenizer:
    """Lossless byte-level tokenizer with optional learned BPE merges."""

    def __init__(self, merges: List[Tuple[int, int]] | None = None):
        self.set_merges(list(merges or []))

    # -------------------------------------------------------------- #
    def set_merges(self, merges: List[Tuple[int, int]]) -> None:
        self.merges: List[Tuple[int, int]] = [
            (int(a), int(b)) for a, b in merges
        ]
        self._merge_ids: Dict[Tuple[int, int], int] = {
            m: FIRST_MERGE_ID + i for i, m in enumerate(self.merges)
        }
        self._id_to_pair: Dict[int, Tuple[int, int]] = {
            v: k for k, v in self._merge_ids.items()
        }
        self.vocab_size = FIRST_MERGE_ID + len(self.merges)

    # -------------------------------------------------------------- #
    def train_merges(self, text: str, num_merges: int) -> List[Tuple[int, int]]:
        """Learn `num_merges` merges greedily from `text` (deterministic)."""
        seq: List[int] = [BYTE_ID_OFFSET + b for b in text.encode("utf-8")]
        learned: List[Tuple[int, int]] = []
        for _ in range(max(0, num_merges)):
            counts: Dict[Tuple[int, int], int] = {}
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
            if not counts:
                break
            # most frequent; tie-break by smaller pair value (deterministic)
            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            new_id = FIRST_MERGE_ID + len(learned)
            seq = _apply_merge(seq, best, new_id)
            learned.append(best)
        self.set_merges(learned)
        return learned

    # -------------------------------------------------------------- #
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = [BYTE_ID_OFFSET + b for b in text.encode("utf-8")]
        for rank, pair in enumerate(self.merges):
            ids = _apply_merge(ids, pair, FIRST_MERGE_ID + rank)
        if add_bos:
            ids = [SPECIAL_TOKENS["<|bos|>"]] + ids
        if add_eos:
            ids = ids + [SPECIAL_TOKENS["<|eos|>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        out_bytes: List[int] = []
        for tok in ids:
            stack: List[int] = [int(tok)]
            while stack:
                x = stack.pop()
                if x in SPECIAL_IDS:
                    continue  # specials decode to nothing
                if x in self._id_to_pair:
                    left, right = self._id_to_pair[x]
                    stack.append(right)
                    stack.append(left)
                elif BYTE_ID_OFFSET <= x < BYTE_ID_OFFSET + NUM_BYTES:
                    out_bytes.append(x - BYTE_ID_OFFSET)
                else:
                    raise ValueError(f"Unknown token id: {x}")
        return bytes(out_bytes).decode("utf-8", errors="replace")

    # -------------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"merges": [[a, b] for a, b in self.merges],
                 "special_tokens": SPECIAL_TOKENS},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(merges=[(a, b) for a, b in data["merges"]])


def _apply_merge(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every adjacent occurrence of `pair` in `seq` by `new_id`."""
    out: List[int] = []
    i, n = 0, len(seq)
    while i < n:
        if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def roundtrip_check(tok: ByteTokenizer, text: str) -> bool:
    """Utility used by tests: encode->decode must reproduce the exact text."""
    return tok.decode(tok.encode(text)) == text
