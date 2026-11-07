import json
from pathlib import Path

from src.ai_provider import FakeProvider
from src.case_store import (
    ai_cache_path,
    get_or_generate_case_ai_risks,
    list_case_files,
    load_case,
    load_cached_ai_risks,
    save_cached_ai_risks,
)
from src.models import ContractContext, RiskItem
from src.risk_generator import GeneratedRisk


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


def _valid_ai_list_response(count: int = 3) -> str:
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


def _write_case(tmp_path, filename: str, nome: str = "Caso de teste") -> Path:
    path = tmp_path / filename
    path.write_text(
        json.dumps(
            {
                "nome": nome,
                "objeto": "Contratação de notebooks",
                "tipo_contratacao": "aquisicao",
                "valor_estimado": 500000,
                "criticidade": "alta",
                "prazo": "6 meses",
                "modalidade": "pregao",
                "contexto": "Renovação do parque de notebooks.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_list_case_files_excludes_ia_cache_files(tmp_path):
    _write_case(tmp_path, "exemplo.json")
    (tmp_path / "exemplo_ia_pt.json").write_text("{}", encoding="utf-8")
    (tmp_path / "exemplo_ia_en.json").write_text("{}", encoding="utf-8")

    files = list_case_files(tmp_path)

    assert [path.name for path in files] == ["exemplo.json"]


def test_list_case_files_returns_empty_for_missing_dir(tmp_path):
    assert list_case_files(tmp_path / "nao_existe") == []


def test_load_case_reads_context_and_nome(tmp_path):
    path = _write_case(tmp_path, "exemplo.json", nome="Notebooks 2026")

    nome, context = load_case(path)

    assert nome == "Notebooks 2026"
    assert context.objeto == "Contratação de notebooks"
    assert context.tipo_contratacao == "aquisicao"
    assert context.valor_estimado == 500000


def test_load_case_falls_back_to_filename_when_nome_missing(tmp_path):
    path = tmp_path / "sem_nome.json"
    path.write_text(json.dumps({"objeto": "x", "tipo_contratacao": "servico"}), encoding="utf-8")

    nome, context = load_case(path)

    assert nome == "sem_nome"
    assert context.objeto == "x"


def test_ai_cache_path_uses_case_filename_stem_and_language(tmp_path):
    path = tmp_path / "exemplo.json"

    assert ai_cache_path(path, "pt", tmp_path) == tmp_path / "exemplo_ia_pt.json"
    assert ai_cache_path(path, "en", tmp_path) == tmp_path / "exemplo_ia_en.json"


def test_load_cached_ai_risks_returns_none_when_missing(tmp_path):
    path = tmp_path / "exemplo.json"

    assert load_cached_ai_risks(path, "pt", tmp_path) is None


def test_save_and_load_cached_ai_risks_roundtrip(tmp_path):
    path = tmp_path / "exemplo.json"
    data = {"planejamento": [GeneratedRisk("T", "C", "Co", 3, 4, "P", "Co2", "R")]}

    save_cached_ai_risks(path, "pt", data, tmp_path)
    loaded = load_cached_ai_risks(path, "pt", tmp_path)

    assert loaded == data


def test_get_or_generate_case_ai_risks_calls_provider_only_once_across_two_runs(tmp_path):
    case_path = _write_case(tmp_path, "exemplo.json")
    risks = [_risk("R001", "planejamento")]
    provider = FakeProvider(_valid_ai_list_response(1))

    first = get_or_generate_case_ai_risks(case_path, "pt", risks, _context(), provider, cases_dir=tmp_path)
    assert len(provider.calls) == 1
    assert first["planejamento"][0].titulo == "Risco de IA 1"

    second = get_or_generate_case_ai_risks(case_path, "pt", risks, _context(), provider, cases_dir=tmp_path)

    assert len(provider.calls) == 1  # nao chamou a IA de novo
    assert second == first


def test_get_or_generate_case_ai_risks_separate_languages_do_not_share_cache(tmp_path):
    case_path = _write_case(tmp_path, "exemplo.json")
    risks = [_risk("R001", "planejamento")]
    provider = FakeProvider(_valid_ai_list_response(1))

    get_or_generate_case_ai_risks(case_path, "pt", risks, _context(), provider, cases_dir=tmp_path)
    get_or_generate_case_ai_risks(case_path, "en", risks, _context(), provider, cases_dir=tmp_path)

    assert len(provider.calls) == 2
    assert (tmp_path / "exemplo_ia_pt.json").exists()
    assert (tmp_path / "exemplo_ia_en.json").exists()
