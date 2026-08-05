"""Output-boundary helpers shared by model-facing workflows.

Readraft defaults to automatic output sizing. An explicit token value is only
sent when an operator configured one or when a provider already proved that
its implicit single-response boundary was too small.
"""

from __future__ import annotations


def optional_output_token_limit(value: object) -> int | None:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def expanded_output_token_limit(
    current: int | None,
    *,
    observed_output_tokens: int = 0,
) -> int:
    """Request more room after a real provider-side length stop.

    There is deliberately no application maximum. The next value is derived
    from the boundary just observed; provider/model capability remains the
    ultimate constraint.
    """

    observed = max(0, int(observed_output_tokens or 0))
    previous = max(0, int(current or 0))
    # A few OpenAI-compatible gateways omit usage on a length stop. In that
    # case 4096 is a recovery request, not a ceiling; later length responses
    # continue doubling without an application maximum.
    baseline = max(previous, observed, 4_096 if not previous and not observed else 1)
    return baseline * 2
