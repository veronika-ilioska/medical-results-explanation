"""Compatibility helpers for vLLM tokenizer loading.

Some vLLM versions cache Hugging Face tokenizers by reading the legacy
``all_special_tokens_extended`` property. Newer Transformers tokenizer
backends, including the generic ``TokenizersBackend``, may not expose that
property even though they still expose ``all_special_tokens``.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def patch_all_special_tokens_extended() -> None:
    """Add the legacy tokenizer property expected by older vLLM releases."""
    try:
        import transformers.convert_slow_tokenizer as convert_slow_tokenizer
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        log.debug("Skipping Transformers tokenizer compatibility patch: %s", exc)
        return

    @property
    def all_special_tokens_extended(self):  # type: ignore[no-untyped-def]
        return self.all_special_tokens

    patched = []
    for cls in (
        PreTrainedTokenizerBase,
        getattr(convert_slow_tokenizer, "TokenizersBackend", None),
    ):
        if cls is not None and not hasattr(cls, "all_special_tokens_extended"):
            cls.all_special_tokens_extended = all_special_tokens_extended
            patched.append(cls.__name__)

    if patched:
        log.info(
            "Patched Transformers tokenizer classes for vLLM compatibility: %s",
            ", ".join(patched),
        )
