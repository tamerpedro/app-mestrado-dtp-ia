import json

import pytest

from evaluation.cases.loader import load_cases_from_dir
from evaluation.cases.provenance import split_by_provenance
from evaluation.cases.schema import CaseFormatError, case_from_dict

VALID_PAYLOAD = {
    "case_id": "case-001",
    "context": {
        "objeto": "Aquisicao de software",
        "tipo_contratacao": "software",
        "valor_estimado": 100000.0,
        "criticidade": "alta",
        "prazo": "12 meses",
        "modalidade": "pregao",
        "contexto": "contexto de teste",
    },
    "gabarito_ids": ["S001", "S002"],
    "used_to_build_base": False,
}


def test_case_from_dict_parses_a_valid_payload():
    case = case_from_dict(VALID_PAYLOAD)

    assert case.case_id == "case-001"
    assert case.context.objeto == "Aquisicao de software"
    assert case.gabarito_ids == ("S001", "S002")
    assert case.used_to_build_base is False
    # Campos opcionais de metodologia: ausentes no payload, devem ficar None.
    assert case.context_predates_risk_map is None
    assert case.annotator_count is None


def test_case_from_dict_parses_optional_methodology_fields_when_present():
    payload = {**VALID_PAYLOAD, "context_predates_risk_map": True, "annotator_count": 2}

    case = case_from_dict(payload)

    assert case.context_predates_risk_map is True
    assert case.annotator_count == 2


@pytest.mark.parametrize("missing_field", ["case_id", "context", "gabarito_ids", "used_to_build_base"])
def test_case_from_dict_rejects_missing_top_level_field(missing_field):
    payload = {key: value for key, value in VALID_PAYLOAD.items() if key != missing_field}
    with pytest.raises(CaseFormatError):
        case_from_dict(payload)


def test_case_from_dict_rejects_missing_context_field():
    payload = json.loads(json.dumps(VALID_PAYLOAD))
    del payload["context"]["objeto"]
    with pytest.raises(CaseFormatError):
        case_from_dict(payload)


def test_case_from_dict_rejects_empty_gabarito():
    payload = json.loads(json.dumps(VALID_PAYLOAD))
    payload["gabarito_ids"] = []
    with pytest.raises(CaseFormatError):
        case_from_dict(payload)


def test_load_cases_from_dir_returns_empty_list_for_missing_directory(tmp_path):
    assert load_cases_from_dir(tmp_path / "does-not-exist") == []


def test_load_cases_from_dir_rejects_duplicate_case_ids(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(VALID_PAYLOAD), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(VALID_PAYLOAD), encoding="utf-8")

    with pytest.raises(CaseFormatError):
        load_cases_from_dir(tmp_path)


def test_split_by_provenance_excludes_cases_used_to_build_the_base(synthetic_risks):
    leaked = case_from_dict({**VALID_PAYLOAD, "case_id": "leaked-1", "used_to_build_base": True})
    clean = case_from_dict({**VALID_PAYLOAD, "case_id": "clean-1", "used_to_build_base": False})

    report = split_by_provenance([leaked, clean], synthetic_risks)

    assert [case.case_id for case in report.main_eval_cases] == ["clean-1"]
    assert [case.case_id for case in report.excluded_leaked_cases] == ["leaked-1"]
    assert report.total_cases == 2


def test_split_by_provenance_reports_gabarito_ids_missing_from_the_library(synthetic_risks):
    payload = {**VALID_PAYLOAD, "case_id": "case-with-gap", "gabarito_ids": ["S001", "DOES-NOT-EXIST"]}
    case = case_from_dict(payload)

    report = split_by_provenance([case], synthetic_risks)

    assert report.cases_with_unknown_gabarito_ids == {"case-with-gap": ("DOES-NOT-EXIST",)}


def test_split_by_provenance_warns_when_methodology_fields_are_missing(synthetic_risks):
    payload = {**VALID_PAYLOAD, "case_id": "case-no-methodology"}
    case = case_from_dict(payload)  # sem context_predates_risk_map nem annotator_count

    report = split_by_provenance([case], synthetic_risks)

    warnings = report.methodology_warnings["case-no-methodology"]
    assert any("context_predates_risk_map" in warning for warning in warnings)
    assert any("annotator_count" in warning for warning in warnings)


def test_split_by_provenance_warns_when_context_does_not_predate_the_risk_map(synthetic_risks):
    payload = {
        **VALID_PAYLOAD,
        "case_id": "case-hindsight",
        "context_predates_risk_map": False,
        "annotator_count": 2,
    }
    case = case_from_dict(payload)

    report = split_by_provenance([case], synthetic_risks)

    warnings = report.methodology_warnings["case-hindsight"]
    assert any("hindsight" in warning or "retrospectiva" in warning for warning in warnings)
    assert not any("annotator_count" in warning for warning in warnings)


def test_split_by_provenance_warns_when_fewer_than_two_annotators(synthetic_risks):
    payload = {
        **VALID_PAYLOAD,
        "case_id": "case-one-annotator",
        "context_predates_risk_map": True,
        "annotator_count": 1,
    }
    case = case_from_dict(payload)

    report = split_by_provenance([case], synthetic_risks)

    warnings = report.methodology_warnings["case-one-annotator"]
    assert any("annotator_count=1" in warning for warning in warnings)


def test_split_by_provenance_has_no_methodology_warnings_for_a_well_formed_case(synthetic_risks):
    payload = {
        **VALID_PAYLOAD,
        "case_id": "case-well-formed",
        "context_predates_risk_map": True,
        "annotator_count": 2,
    }
    case = case_from_dict(payload)

    report = split_by_provenance([case], synthetic_risks)

    assert "case-well-formed" not in report.methodology_warnings
