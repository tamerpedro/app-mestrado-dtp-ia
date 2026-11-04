"""Bootstrap pareado por caso, para comparar o motor atual com os demais.

Para cada caso, calcula-se a diferenca de uma metrica entre o motor
candidato e o motor de referencia (``motor atual de regras``). O bootstrap
pareado reamostra os *casos* (mantendo o pareamento) para estimar um
intervalo de confianca para a diferenca media -- sem pressupor normalidade
e respeitando que as duas amostras nao sao independentes (mesmo caso).

Implementado so com a biblioteca padrao (``random``, ``statistics``) para
nao exigir numpy no pacote principal da avaliacao.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedBootstrapResult:
    n_cases: int
    observed_mean_diff: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_resamples: int
    seed: int
    p_value_two_sided: float
    """Fracao de reamostras cuja diferenca media cruza para o lado oposto de zero
    em relacao ao valor observado -- uma aproximacao bootstrap simples do p-valor,
    nao um teste exato."""


def paired_bootstrap_ci(
    diffs: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedBootstrapResult:
    if len(diffs) < 2:
        raise ValueError("Bootstrap pareado precisa de pelo menos 2 casos com a metrica calculada.")

    rng = random.Random(seed)
    n = len(diffs)
    observed_mean = statistics.fmean(diffs)

    resampled_means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(statistics.fmean(sample))
    resampled_means.sort()

    alpha = 1 - confidence
    lower_index = int((alpha / 2) * n_resamples)
    upper_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))

    if observed_mean >= 0:
        crossings = sum(1 for mean in resampled_means if mean <= 0)
    else:
        crossings = sum(1 for mean in resampled_means if mean >= 0)
    p_value = min(1.0, 2 * crossings / n_resamples)

    return PairedBootstrapResult(
        n_cases=n,
        observed_mean_diff=observed_mean,
        ci_lower=resampled_means[lower_index],
        ci_upper=resampled_means[upper_index],
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        p_value_two_sided=p_value,
    )
