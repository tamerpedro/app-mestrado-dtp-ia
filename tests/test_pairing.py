import src.pairing as pairing_module
from src.ai_provider import FakeProvider
from src.models import ContractContext, RiskItem
from src.pairing import ORIGIN_AI, ORIGIN_LIBRARY, build_category_pairs, top_library_risk_per_category


def _context() -> ContractContext:
    return ContractContext(
        objeto="Contratação de solução de TIC",
        tipo_contratacao="software",
        valor_estimado=100000,
        criticidade="alta",
        prazo="12 meses",
        modalidade="pregao",
        contexto="Contexto de teste",
    )


def _risk(risk_id: str, categoria: str) -> RiskItem:
    return RiskItem(
        id=risk_id,
        titulo=f"Risco {risk_id}",
        categoria=categoria,
        tipo_contratacao=["software"],
        palavras_chave=[],
        causa="Causa",
        consequencia="Consequencia",
        probabilidade_padrao=3,
        impacto_padrao=4,
        acao_preventiva="Prevenir",
        acao_contingencia="Conter",
        responsavel_sugerido="Equipe",
    )


def _valid_ai_response() -> str:
    return (
        '{"titulo": "Risco de IA", "causa": "Causa IA", "consequencia": "Consequencia IA", '
        '"probabilidade_padrao": 3, "impacto_padrao": 4, "acao_preventiva": "Prevenir IA", '
        '"acao_contingencia": "Conter IA", "responsavel_sugerido": "Equipe IA"}'
    )


def test_top_library_risk_per_category_respects_cutoff(monkeypatch):
    risk_above = _risk("R001", "planejamento")
    risk_below = _risk("R002", "selecao")
    scores = {"R001": 5, "R002": 1}
    monkeypatch.setattr(pairing_module, "suggestion_score", lambda risk, context: scores[risk.id])

    chosen = top_library_risk_per_category([risk_above, risk_below], _context(), minimum_score=2)

    assert chosen == {"planejamento": risk_above}


def test_build_category_pairs_creates_pair_with_both_origins(monkeypatch):
    risk_above = _risk("R001", "planejamento")
    monkeypatch.setattr(pairing_module, "suggestion_score", lambda risk, context: 5)
    provider = FakeProvider(_valid_ai_response())

    result = build_category_pairs([risk_above], _context(), provider, minimum_score=2, seed=1)

    pairs = {pair.categoria: pair for pair in result.pairs}
    assert "planejamento" in pairs
    pair = pairs["planejamento"]
    assert pair.sem_par is False
    origins = {s.origin for s in pair.suggestions}
    assert origins == {ORIGIN_LIBRARY, ORIGIN_AI}
    slot_ids = [s.slot_id for s in pair.suggestions]
    assert all(slot.startswith("S") for slot in slot_ids)
    assert len(set(slot_ids)) == len(slot_ids)


def test_build_category_pairs_marks_sem_par_when_no_library_risk_above_cutoff(monkeypatch):
    risk_low_score = _risk("R002", "selecao")
    monkeypatch.setattr(pairing_module, "suggestion_score", lambda risk, context: 0)
    provider = FakeProvider(_valid_ai_response())

    result = build_category_pairs([risk_low_score], _context(), provider, minimum_score=2, seed=1)

    pairs = {pair.categoria: pair for pair in result.pairs}
    assert "selecao" in pairs
    pair = pairs["selecao"]
    assert pair.sem_par is True
    assert len(pair.suggestions) == 1
    assert pair.suggestions[0].origin == ORIGIN_AI


def test_build_category_pairs_skips_category_when_ai_fails_and_no_library_risk(monkeypatch):
    monkeypatch.setattr(pairing_module, "suggestion_score", lambda risk, context: 0)
    provider = FakeProvider(fail=True)
    risk = _risk("R003", "gestao")

    result = build_category_pairs([risk], _context(), provider, minimum_score=2, seed=1)

    pairs = {pair.categoria: pair for pair in result.pairs}
    assert "gestao" not in pairs


def test_library_suggestion_keeps_original_id_and_ai_suggestion_has_no_id(monkeypatch):
    risk_above = _risk("R001", "planejamento")
    monkeypatch.setattr(pairing_module, "suggestion_score", lambda risk, context: 5)
    provider = FakeProvider(_valid_ai_response())

    result = build_category_pairs([risk_above], _context(), provider, minimum_score=2, seed=1)

    by_origin = {s.origin: s for s in result.pairs[0].suggestions}
    assert by_origin[ORIGIN_LIBRARY].row.id == "R001"
    assert by_origin[ORIGIN_AI].row.id == ""


def test_build_category_pairs_is_deterministic_for_a_given_seed(monkeypatch):
    risk_above = _risk("R001", "planejamento")
    monkeypatch.setattr(pairing_module, "suggestion_score", lambda risk, context: 5)
    provider = FakeProvider(_valid_ai_response())

    result_a = build_category_pairs([risk_above], _context(), provider, minimum_score=2, seed=42)
    result_b = build_category_pairs([risk_above], _context(), provider, minimum_score=2, seed=42)

    origins_a = [s.origin for pair in result_a.pairs for s in pair.suggestions]
    origins_b = [s.origin for pair in result_b.pairs for s in pair.suggestions]
    assert origins_a == origins_b
    assert result_a.seed == 42 == result_b.seed


def test_build_category_pairs_calls_on_ai_failure_callback(monkeypatch):
    risk_above = _risk("R001", "planejamento")
    monkeypatch.setattr(pairing_module, "suggestion_score", lambda risk, context: 5)
    provider = FakeProvider(fail=True)
    failures = []

    result = build_category_pairs(
        [risk_above], _context(), provider, minimum_score=2, seed=1, on_ai_failure=lambda cat, err: failures.append((cat, err))
    )

    assert failures and failures[0][0] == "planejamento"
    pair = result.pairs[0]
    assert pair.sem_par is False
    assert len(pair.suggestions) == 1
    assert pair.suggestions[0].origin == ORIGIN_LIBRARY
