"""Garante que o exemplo versionado em evaluation/cases/synthetic/ continua
valido e que a pasta de casos reais (gitignored) carrega vazia num checkout
limpo -- ambos fazem parte do contrato do protocolo documentado no README.
"""

from evaluation.cases.loader import load_real_cases, load_synthetic_cases


def test_synthetic_example_case_loads_and_is_well_formed():
    cases = load_synthetic_cases()

    assert len(cases) == 1
    (case,) = cases
    assert case.case_id == "synthetic-001"
    assert case.used_to_build_base is False
    assert case.gabarito_ids


def test_real_cases_directory_is_empty_in_a_clean_checkout():
    assert load_real_cases() == []
