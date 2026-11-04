"""Reprodutibilidade: roda o mesmo motor repetidamente sobre o mesmo caso e
compara os rankings obtidos.

Motores deterministicos (regras, BM25, embeddings com modelo fixo sem
amostragem) devem ter reprodutibilidade exata (1.0). O motor LLM, se usado
com ``temperature > 0`` ou um servidor que nao fixa seed, pode variar --
e exatamente isso que esta metrica evidencia.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReproducibilityResult:
    exact_match_rate: float
    """Fracao das repeticoes (a partir da 2a) cujo ranking completo e identico, posicao a
    posicao, ao da primeira execucao."""
    mean_top_k_jaccard: float
    """Media da similaridade de Jaccard entre o top-k da 1a execucao e o das demais
    (metrica mais tolerante: ignora a ordem dentro do top-k)."""
    runs: tuple[tuple[str, ...], ...]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def measure_reproducibility(
    rank_fn: Callable[[], Sequence[str]],
    repeats: int,
    top_k: int = 10,
) -> ReproducibilityResult:
    if repeats < 2:
        raise ValueError("repeats precisa ser >= 2 para medir reprodutibilidade.")

    runs = tuple(tuple(rank_fn()) for _ in range(repeats))
    baseline = runs[0]

    exact_matches = sum(1 for run in runs[1:] if run == baseline)
    jaccards = [_jaccard(baseline[:top_k], run[:top_k]) for run in runs[1:]]

    return ReproducibilityResult(
        exact_match_rate=exact_matches / (repeats - 1),
        mean_top_k_jaccard=sum(jaccards) / len(jaccards),
        runs=runs,
    )
