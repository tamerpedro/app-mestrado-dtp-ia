"""Taxa de IDs invalidos devolvidos por um motor (relevante sobretudo para o LLM)."""

from __future__ import annotations

from collections.abc import Sequence


def invalid_id_rate(raw_ids: Sequence[str], valid_ids: frozenset[str]) -> float | None:
    if not raw_ids:
        return None
    invalid = sum(1 for risk_id in raw_ids if risk_id not in valid_ids)
    return invalid / len(raw_ids)
