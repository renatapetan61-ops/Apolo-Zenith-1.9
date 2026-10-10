"""Audio stack — PLANNED / NOT IMPLEMENTED.

Per the master spec (§11): audio generation requires data, weights and
compute that this environment does not have. We deliberately provide NO
placeholder that could be mistaken for a working TTS/music model, and no
static audio file is ever attached to videos claiming to be "native audio".

Required to implement (documented in docs/SCALING_REQUIREMENTS.md):
  - Mel/neural codec encoder (e.g. EnCodec-class, licensed weights only)
  - Text-to-audio transformer conditioned on text + video timing
  - Licensed speech/music datasets with provenance records
  - GPU training budget

Status: PLANNED. Anything claiming audio generation must fail loudly here.
"""

from __future__ import annotations


class AudioNotAvailable(RuntimeError):
    """Raised when audio generation is requested but unimplemented."""


def require_audio() -> None:
    raise AudioNotAvailable(
        "Audio generation is PLANNED, not implemented. Do not attach static "
        "audio to videos and call it native audio."
    )
