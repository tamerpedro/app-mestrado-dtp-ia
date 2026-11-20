import json

import pytest

from src.ai_provider import FakeProvider
from src.catalog_mix import (
    ORIGIN_AI,
    ORIGIN_CATALOG,
    ai_count_for_category,
    build_category_mix,
    filter_catalog_by_contract_type,
    generate_ai_risks_for_all_categories,
)
from src.models import ContractContext, RiskItem


def _context(tipo: str = "software") -> ContractContext:
    return ContractContext(
        objeto="Contratação de solução de TIC",
        tipo_contratacao=tipo,
        valor_estimado=100000,
        criticidade="alta",
        prazo="12 meses",
        modalidade="pregao",
        contexto="Contexto de teste",
    )


def _risk(risk_id: str, categoria: str, tipo_contratacao: list[str]) -> RiskItem:
    return RiskItem(
        id=risk_id,
        titulo=f"Risco {risk_id}",
        categoria=categoria,
        tipo_contratacao=tipo_contratacao,
        palavras_chave=[],
        causa="Causa",
        consequencia="Consequencia",
        probabilidade_padrao=3,
        impacto_padrao=4,
        acao_preventiva="Prevenir",
        acao_contingencia="Conter",
        responsavel_sugerido="Equipe",
    )


def _valid_ai_list_response(count: int) -> str:
    return json.dumps(
        [
            {
                "titulo": f"Risco de IA {index}",
                "causa": "Causa IA",
                "consequencia": "Consequencia IA",
                "probabilidade_padrao": 3,
                "impacto_padrao": 4,
                "acao_preventiva": "Prevenir IA",
                "acao_contingencia": "Conter IA",
                "responsavel_sugerido": "Equipe IA",
            }
            for index in range(1, count + 1)
        ]
    )


@pytest.mark.parametrize(
    ("total", "expected"),
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (10, 3)],
)
def test_ai_count_for_category_is_min_3_n(total, expected):
    assert ai_count_for_category(total) == expected


def test_filter_catalog_by_contract_type_keeps_matching_only():
    risks = [
        _risk("R001", "planejamento", ["software"]),
        _risk("R002", "planejamento", ["servico"]),
        _risk("R003", "planejamento", ["software", "aquisicao"]),
    ]

    filtered = filter_catalog_by_contract_type(risks, "software")

    assert {risk.id for risk in filtered} == {"R001", "R003"}


def test_filter_catalog_by_contract_type_is_not_a_score_just_inclusion():
    risks = [_risk("R001", "planejamento", ["software"])]

    assert filter_catalog_by_contract_type(risks, "servico") == []
    assert filter_catalog_by_contract_type(risks, "software") == risks


def test_generate_ai_risks_for_all_categories_counts_before_filter(monkeypatch):
    # 3 riscos de "planejamento" no catalogo total, so 1 bate com "software":
    # a contagem de IA usa o total (3), nao o filtrado (1).
    risks = [
        _risk("R001", "planejamento", ["software"]),
        _risk("R002", "planejamento", ["servico"]),
        _risk("R003", "planejamento", ["servico"]),
    ]
    provider = FakeProvider(_valid_ai_list_response(3))

    result = generate_ai_risks_for_all_categories(risks, _context("software"), provider, lang="pt")

    assert len(result["planejamento"]) == 3


def test_generate_ai_risks_for_all_categories_skips_empty_category():
    provider = FakeProvider(_valid_ai_list_response(3))

    result = generate_ai_risks_for_all_categories([], _context(), provider, lang="pt")

    assert all(generated == [] for generated in result.values())
    assert provider.calls == []


def test_generate_ai_risks_for_all_categories_failure_is_isolated_per_category(monkeypatch):
    risks = [_risk("R001", "planejamento", ["software"]), _risk("R002", "selecao", ["software"])]
    provider = FakeProvider(fail=True)
    failures = []

    result = generate_ai_risks_for_all_categories(
        risks, _context(), provider, lang="pt", on_ai_failure=lambda cat, err: failures.append(cat)
    )

    assert result["planejamento"] == []
    assert result["selecao"] == []
    assert set(failures) == {"planejamento", "selecao"}


def test_build_category_mix_combines_catalog_and_ai_items():
    risks = [
        _risk("R001", "planejamento", ["software"]),
        _risk("R002", "planejamento", ["software"]),
    ]
    ai_risks = generate_ai_risks_for_all_categories(risks, _context(), FakeProvider(_valid_ai_list_response(2)), lang="pt")

    result = build_category_mix(risks, _context(), ai_risks, lang="pt", seed=1)

    categories = {c.categoria: c for c in result.categories}
    assert "planejamento" in categories
    origins = {item.origin for item in categories["planejamento"].items}
    assert origins == {ORIGIN_CATALOG, ORIGIN_AI}
    assert len(categories["planejamento"].items) == 4  # 2 catalogo + 2 IA


def test_build_category_mix_assigns_neutral_slot_ids_and_hides_catalog_id_for_ai():
    risks = [_risk("R001", "planejamento", ["software"])]
    ai_risks = {"planejamento": [], "selecao": [], "gestao": [], "solucao": [], "instalacao": [], "cronograma": []}

    result = build_category_mix(risks, _context(), ai_risks, lang="pt", seed=1)

    item = result.categories[0].items[0]
    assert item.slot_id.startswith("S")
    assert item.row.id == "R001"  # catalogo mantem o id original (uso interno, nunca mostrado)
    assert not item.row.justificativa


def test_build_category_mix_skips_category_with_nothing_to_show():
    result = build_category_mix([], _context(), {}, lang="pt", seed=1)

    assert result.categories == []


def test_build_category_mix_is_deterministic_for_a_given_seed():
    risks = [_risk(f"R{i:03d}", "planejamento", ["software"]) for i in range(1, 6)]
    ai_risks = {"planejamento": []}

    result_a = build_category_mix(risks, _context(), ai_risks, lang="pt", seed=42)
    result_b = build_category_mix(risks, _context(), ai_risks, lang="pt", seed=42)

    order_a = [item.row.id for item in result_a.categories[0].items]
    order_b = [item.row.id for item in result_b.categories[0].items]
    assert order_a == order_b
    assert result_a.seed == 42 == result_b.seed


def test_build_category_mix_uses_unique_slot_ids_across_categories():
    risks = [
        _risk("R001", "planejamento", ["software"]),
        _risk("R002", "selecao", ["software"]),
    ]

    result = build_category_mix(risks, _context(), {}, lang="pt", seed=1)

    all_slot_ids = [item.slot_id for category in result.categories for item in category.items]
    assert len(all_slot_ids) == len(set(all_slot_ids))
