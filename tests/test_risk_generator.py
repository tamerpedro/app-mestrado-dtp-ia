import json

import pytest

from src.ai_provider import FakeProvider
from src.models import ContractContext, RiskItem
from src.risk_generator import (
    RiskGenerationError,
    build_prompt,
    build_prompt_v2,
    generate_risk_for_category,
    generate_risks_for_category,
    validate_generated_risk,
    validate_generated_risks_list,
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


# --------------------------------------------------------------------------
# v2: lista de N riscos por categoria, cientes de todo o catalogo
# --------------------------------------------------------------------------


def _valid_ai_list_payload(count: int = 2) -> list[dict]:
    return [
        {
            "titulo": f"Risco gerado {index}",
            "causa": f"Causa gerada {index}",
            "consequencia": f"Consequencia gerada {index}",
            "probabilidade_padrao": 3,
            "impacto_padrao": 4,
            "acao_preventiva": f"Ação preventiva {index}",
            "acao_contingencia": f"Ação de contingência {index}",
            "responsavel_sugerido": "Equipe gerada",
        }
        for index in range(1, count + 1)
    ]


def test_build_prompt_v2_includes_all_catalog_titles_and_generic_instruction():
    titles = ["Risco A da categoria X", "Risco B da categoria Y"]

    prompt = build_prompt_v2(_context(), "Planejamento", 3, titles, [_risk()], lang="pt")

    assert "Risco A da categoria X" in prompt
    assert "Risco B da categoria Y" in prompt
    assert "genérico" in prompt or "genéric" in prompt
    assert "Não repita" in prompt or "não repita" in prompt.lower()
    assert "3" in prompt


def test_build_prompt_v2_requests_exact_quantity():
    prompt = build_prompt_v2(_context(), "Planejamento", 5, [], [], lang="pt")

    assert "5 riscos" in prompt


def test_generate_risks_for_category_parses_valid_json_list():
    provider = FakeProvider(json.dumps(_valid_ai_list_payload(3)))

    generated = generate_risks_for_category(
        provider, _context(), "Planejamento", 3, ["Risco existente"], [_risk()], lang="pt"
    )

    assert len(generated) == 3
    assert generated[0].titulo == "Risco gerado 1"
    assert len(provider.calls) == 1


def test_generate_risks_for_category_parses_fenced_json_list():
    raw_response = "```json\n" + json.dumps(_valid_ai_list_payload(2)) + "\n```"
    provider = FakeProvider(raw_response)

    generated = generate_risks_for_category(provider, _context(), "Planejamento", 2, [], [], lang="pt")

    assert len(generated) == 2


def test_validate_generated_risks_list_rejects_empty_list():
    with pytest.raises(RiskGenerationError):
        validate_generated_risks_list([])


def test_validate_generated_risks_list_rejects_invalid_item():
    payload = _valid_ai_list_payload(1)
    payload[0]["titulo"] = "   "

    with pytest.raises(RiskGenerationError):
        validate_generated_risks_list(payload)


def test_generate_risks_for_category_raises_on_non_list_json():
    provider = FakeProvider(json.dumps({"titulo": "nao e uma lista"}))

    with pytest.raises(RiskGenerationError):
        generate_risks_for_category(provider, _context(), "Planejamento", 1, [], [], lang="pt")


def test_generate_risks_for_category_raises_when_provider_fails():
    provider = FakeProvider(fail=True)

    with pytest.raises(RiskGenerationError):
        generate_risks_for_category(provider, _context(), "Planejamento", 2, [], [], lang="pt")
