import json

import pytest

from src.ai_provider import FakeProvider
from src.models import ContractContext, RiskItem
from src.risk_generator import (
    RiskGenerationError,
    build_prompt,
    generate_risk_for_category,
    validate_generated_risk,
)


def _context() -> ContractContext:
    return ContractContext(
        objeto="Contratação de solução de monitoramento de rede",
        tipo_contratacao="software",
        valor_estimado=500000,
        criticidade="alta",
        prazo="12 meses",
        modalidade="pregao",
        contexto="Ambiente com múltiplos datacenters e alta exigência de disponibilidade.",
    )


def _risk(risk_id: str = "R001", titulo: str = "Risco de biblioteca") -> RiskItem:
    return RiskItem(
        id=risk_id,
        titulo=titulo,
        categoria="planejamento",
        tipo_contratacao=["software"],
        palavras_chave=["rede"],
        causa="Causa de exemplo",
        consequencia="Consequencia de exemplo",
        probabilidade_padrao=3,
        impacto_padrao=4,
        acao_preventiva="Prevenir",
        acao_contingencia="Conter",
        responsavel_sugerido="Equipe técnica",
    )


def _valid_payload() -> dict:
    return {
        "titulo": "Risco gerado pela IA",
        "causa": "Causa gerada",
        "consequencia": "Consequencia gerada",
        "probabilidade_padrao": 3,
        "impacto_padrao": 4,
        "acao_preventiva": "Ação preventiva gerada",
        "acao_contingencia": "Ação de contingência gerada",
        "responsavel_sugerido": "Equipe gerada",
    }


def test_build_prompt_includes_context_category_and_instruction_to_differ():
    prompt = build_prompt(_context(), "Planejamento", _risk(), [_risk("R002", "Outro risco")], lang="pt")

    assert "Contratação de solução de monitoramento de rede" in prompt
    assert "Planejamento" in prompt
    assert "Risco de biblioteca" in prompt
    assert "DIFERENTE" in prompt


def test_build_prompt_without_library_risk_explains_there_is_no_reference():
    prompt = build_prompt(_context(), "Planejamento", None, [_risk()], lang="pt")

    assert "Nenhum risco da biblioteca atingiu a pontuação mínima" in prompt


def test_build_prompt_truncates_long_example_fields():
    long_risk = _risk(titulo="x" * 500)

    prompt = build_prompt(_context(), "Planejamento", None, [long_risk], lang="pt")

    assert "x" * 500 not in prompt
    assert "…" in prompt


def test_validate_generated_risk_accepts_well_formed_payload():
    generated = validate_generated_risk(_valid_payload())

    assert generated.titulo == "Risco gerado pela IA"
    assert generated.probabilidade_padrao == 3
    assert generated.impacto_padrao == 4


@pytest.mark.parametrize("missing_field", ["titulo", "causa", "acao_preventiva", "responsavel_sugerido"])
def test_validate_generated_risk_rejects_missing_text_field(missing_field):
    payload = _valid_payload()
    payload[missing_field] = "   "

    with pytest.raises(RiskGenerationError):
        validate_generated_risk(payload)


@pytest.mark.parametrize("value", [0, 6, "alta", None])
def test_validate_generated_risk_rejects_invalid_scale(value):
    payload = _valid_payload()
    payload["probabilidade_padrao"] = value

    with pytest.raises(RiskGenerationError):
        validate_generated_risk(payload)


def test_generate_risk_for_category_parses_fenced_json_response():
    raw_response = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    provider = FakeProvider(raw_response)

    generated = generate_risk_for_category(provider, _context(), "Planejamento", _risk(), [_risk("R002")], lang="pt")

    assert generated.titulo == "Risco gerado pela IA"
    assert len(provider.calls) == 1


def test_generate_risk_for_category_raises_on_invalid_json():
    provider = FakeProvider("isto nao e json")

    with pytest.raises(RiskGenerationError):
        generate_risk_for_category(provider, _context(), "Planejamento", _risk(), [], lang="pt")


def test_generate_risk_for_category_raises_when_provider_fails():
    provider = FakeProvider(fail=True)

    with pytest.raises(RiskGenerationError):
        generate_risk_for_category(provider, _context(), "Planejamento", _risk(), [], lang="pt")
