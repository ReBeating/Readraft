from __future__ import annotations

import re


def effective_char_count(text: str) -> int:
    """Count visible characters without whitespace."""

    return len(re.sub(r"\s+", "", text))
