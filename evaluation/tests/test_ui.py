"""Testa a logica pura de evaluation/ui.py (funcoes que nao chamam st.*).

Nao testamos as funcoes render_*/main() aqui -- elas sao a camada de
apresentacao Streamlit e dependem de um runtime de app (ScriptRunContext)
que nao existe sob pytest. A validacao e persistencia (o que de fato
importa testar) ficam isoladas em funcoes puras, testadas abaixo.
"""

from __future__ import annotations

import json

import pytest

from evaluation.cases.loader import load_cases_from_dir
from evaluation.cases.schema import CaseFormatError, case_from_dict
from evaluation.config import LLMConfigError
from evaluation.ui import (
    build_run_config,
    case_file_path,
    case_to_draft,
    compute_found_missing,
    draft_to_payload,
    filter_risks,
    load_existing_cases,
    new_draft_case,
    save_case,
)


def _filled_draft(**overrides) -> dict:
    draft = new_draft_case()
    draft.update(
        {
            "case_id": "case-001",
            "objeto": "Aquisicao de software de seguranca",
            "tipo_contratacao": "software",
            "valor_estimado": 100000.0,
            "criticidade": "alta",
            "prazo": "12 meses",
            "modalidade": "pregao",
            "contexto": "contexto de teste",
            "gabarito_ids": ["S001", "S002"],
        }
    )
    draft.update(overrides)
    return draft


def test_new_draft_case_has_empty_gabarito_and_no_methodology_info():
    draft = new_draft_case()

    assert draft["case_id"] == ""
    assert draft["gabarito_ids"] == []
    assert draft["context_predates_risk_map"] is None
    assert draft["annotator_count"] is None


def test_draft_to_payload_omits_optional_fields_when_not_informed():
    payload = draft_to_payload(_filled_draft())

    assert "context_predates_risk_map" not in payload
    assert "annotator_count" not in payload
    assert payload["gabarito_ids"] == ["S001", "S002"]
    assert payload["context"]["objeto"] == "Aquisicao de software de seguranca"


def test_draft_to_payload_includes_optional_fields_when_informed():
    draft = _filled_draft(context_predates_risk_map=True, annotator_count=2)

    payload = draft_to_payload(draft)

    assert payload["context_predates_risk_map"] is True
    assert payload["annotator_count"] == 2


def test_draft_to_payload_round_trips_through_the_real_schema():
    """draft_to_payload nao deve inventar um formato proprio -- o resultado
    precisa ser aceito por cases.schema.case_from_dict sem ajustes."""
    draft = _filled_draft(context_predates_risk_map=False, annotator_count=1)

    case = case_from_dict(draft_to_payload(draft))

    assert case.case_id == "case-001"
    assert case.gabarito_ids == ("S001", "S002")
    assert case.context_predates_risk_map is False
    assert case.annotator_count == 1


def test_case_to_draft_is_the_inverse_of_draft_to_payload():
    draft = _filled_draft(context_predates_risk_map=True, annotator_count=2, provenance_notes="nota")
    case = case_from_dict(draft_to_payload(draft))

    rebuilt_draft = case_to_draft(case)

    assert rebuilt_draft == draft


def test_case_file_path_sanitizes_unsafe_characters(tmp_path):
    path = case_file_path("caso 1 / teste?", cases_dir=tmp_path)

    assert path.parent == tmp_path
    assert path.name == "caso1teste.json"
    assert "/" not in path.name
    assert " " not in path.name


def test_save_case_rejects_empty_case_id(tmp_path):
    ok, message = save_case(_filled_draft(case_id=""), cases_dir=tmp_path)

    assert ok is False
    assert "case_id" in message
    assert list(tmp_path.glob("*.json")) == []


def test_save_case_rejects_empty_gabarito_without_writing_a_file(tmp_path):
    ok, message = save_case(_filled_draft(gabarito_ids=[]), cases_dir=tmp_path)

    assert ok is False
    assert list(tmp_path.glob("*.json")) == []


def test_save_case_persists_a_valid_case_loadable_by_the_real_loader(tmp_path):
    ok, message = save_case(_filled_draft(), cases_dir=tmp_path)

    assert ok is True
    assert "case-001" in message

    loaded = load_cases_from_dir(tmp_path)
    assert [case.case_id for case in loaded] == ["case-001"]
    assert loaded[0].gabarito_ids == ("S001", "S002")


def test_save_case_overwrites_the_same_case_id_file(tmp_path):
    save_case(_filled_draft(provenance_notes="primeira versao"), cases_dir=tmp_path)
    save_case(_filled_draft(provenance_notes="segunda versao"), cases_dir=tmp_path)

    assert len(list(tmp_path.glob("*.json"))) == 1
    (loaded,) = load_cases_from_dir(tmp_path)
    assert loaded.provenance_notes == "segunda versao"


def test_load_existing_cases_returns_empty_list_for_a_fresh_directory(tmp_path):
    cases, error = load_existing_cases(tmp_path / "does-not-exist-yet")

    assert cases == []
    assert error is None


def test_load_existing_cases_surfaces_the_real_loader_error(tmp_path):
    (tmp_path / "broken.json").write_text(json.dumps({"case_id": "x"}), encoding="utf-8")

    cases, error = load_existing_cases(tmp_path)

    assert cases == []
    assert error is not None
    assert "x" in error or "broken" in error.lower() or "obrigat" in error.lower()


def test_filter_risks_matches_id_title_or_category(synthetic_risks):
    assert [r.id for r in filter_risks(synthetic_risks, "")] == [r.id for r in synthetic_risks]
    assert [r.id for r in filter_risks(synthetic_risks, "s001")] == ["S001"]
    assert [r.id for r in filter_risks(synthetic_risks, "seguranca")] == ["S001"]
    assert [r.id for r in filter_risks(synthetic_risks, "planejamento")] == ["S001", "S003", "S004"]
    assert filter_risks(synthetic_risks, "risco-que-nao-existe") == []


def test_compute_found_missing_splits_gabarito_by_presence_in_top_10():
    gabarito = frozenset({"R001", "R002", "R003"})

    found, missing = compute_found_missing(["R001", "R999", "R003"], gabarito)

    assert found == ["R001", "R003"]
    assert missing == ["R002"]


def test_compute_found_missing_never_reports_ids_outside_the_gabarito():
    found, missing = compute_found_missing(["R001", "R999"], frozenset({"R001"}))

    assert found == ["R001"]
    assert missing == []


def test_build_run_config_defaults_keep_llm_disabled_and_local():
    config = build_run_config(
        engines=("rules", "bm25"),
        embedding_backend="hashing",
        llm_enabled=False,
        llm_endpoint="http://localhost:11434/v1/chat/completions",
        llm_model="llama3.1:8b-instruct",
        llm_allow_external=False,
    )

    assert config.engines == ("rules", "bm25")
    assert config.embedding.backend == "hashing"
    assert config.llm.enabled is False


def test_build_run_config_rejects_non_local_llm_endpoint_without_allow_external():
    with pytest.raises(LLMConfigError):
        build_run_config(
            engines=("rules", "llm"),
            embedding_backend="hashing",
            llm_enabled=True,
            llm_endpoint="https://api.example.com/v1/chat/completions",
            llm_model="gpt-4o-mini",
            llm_allow_external=False,
        )


def test_build_run_config_allows_non_local_llm_endpoint_when_explicitly_opted_in():
    config = build_run_config(
        engines=("rules", "llm"),
        embedding_backend="hashing",
        llm_enabled=True,
        llm_endpoint="https://api.example.com/v1/chat/completions",
        llm_model="gpt-4o-mini",
        llm_allow_external=True,
    )

    assert config.llm.endpoint == "https://api.example.com/v1/chat/completions"
