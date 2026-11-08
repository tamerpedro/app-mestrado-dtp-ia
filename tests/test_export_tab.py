"""Aba Exportacao (Parte 3): resultado da avaliacao, formulario final e gravacao do log."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.ai_provider as ai_provider_module
import src.risk_library as risk_library_module
from src.ai_provider import FakeProvider
from src.eval_log import save_evaluation_log

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _valid_ai_risk(suffix: str) -> dict:
    return {
        "titulo": f"Risco gerado por IA {suffix}",
        "causa": "Causa gerada por IA",
        "consequencia": "Consequencia gerada por IA",
        "probabilidade_padrao": 3,
        "impacto_padrao": 3,
        "acao_preventiva": "Acao preventiva gerada por IA",
        "acao_contingencia": "Acao de contingencia gerada por IA",
        "responsavel_sugerido": "Equipe de TI",
    }


def _ai_response(prompt: str) -> str:
    return json.dumps([_valid_ai_risk(str(i)) for i in range(3)])


@pytest.fixture(autouse=True)
def _working_ai_provider(monkeypatch):
    """Substitui o provedor real por um FakeProvider que sempre gera riscos
    validos, para a lista cega ter itens de origem IA (sem rede)."""
    monkeypatch.setattr(ai_provider_module, "build_provider", lambda config: FakeProvider(response=_ai_response))


@pytest.fixture
def _log_dir(tmp_path, monkeypatch):
    """Redireciona a gravacao do log para um diretorio temporario, sem tocar
    data/avaliacao/ do repositorio."""
    directory = tmp_path / "avaliacao"

    def _fake_save(log, directory=None, session_id=None):  # noqa: ARG001 - directory do app ignorado de proposito
        return save_evaluation_log(log, directory=directory_holder["path"], session_id=session_id)

    directory_holder = {"path": directory}
    monkeypatch.setattr("src.eval_log.save_evaluation_log", _fake_save)
    return directory


@pytest.fixture
def _candidates_path(tmp_path, monkeypatch):
    """Redireciona a gravacao de candidatos_ia.csv para um arquivo temporario."""
    path = tmp_path / "candidatos_ia.csv"
    original = risk_library_module.save_matrix_row_to_library

    def _fake_save(_path, row, context, lang="pt"):
        return original(path, row, context, lang=lang)

    monkeypatch.setattr(risk_library_module, "save_matrix_row_to_library", _fake_save)
    return path


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def _generate(at: AppTest) -> AppTest:
    at.button(key="btn_generate_suggestions").click()
    at.run()
    assert not at.exception, at.exception
    return at


def _inclusion_checkboxes(at: AppTest):
    return [checkbox for checkbox in at.checkbox if checkbox.key and checkbox.key.startswith("incl_")]


def _slot_origin(at: AppTest) -> dict[str, str]:
    mix_result = at.session_state["mix_result"]
    return {item.slot_id: item.origin for category in mix_result.categories for item in category.items}


def _choose_one_catalog_and_one_ai_item(at: AppTest) -> AppTest:
    origin = _slot_origin(at)
    checkboxes = _inclusion_checkboxes(at)
    catalog_box = next(cb for cb in checkboxes if origin.get(cb.key.removeprefix("incl_")) == "catalogo")
    ai_box = next(cb for cb in checkboxes if origin.get(cb.key.removeprefix("incl_")) == "ia")
    catalog_box.set_value(True)
    ai_box.set_value(True)
    at.run()
    assert not at.exception, at.exception
    return at


def _fill_mandatory_final_form(at: AppTest, *, codigo: str = "PROC-001") -> AppTest:
    at.text_input(key="codigo_processo").input(codigo)
    at.run()
    at.radio(key="experiencia").set_value("2_a_5")
    at.run()
    assert not at.exception, at.exception
    return at


def test_result_section_shows_composition_metrics_and_reveal_table():
    at = _choose_one_catalog_and_one_ai_item(_generate(_run()))

    reveal_values = [row for df in at.dataframe for row in df.value.to_dict("records")]
    assert any("Catálogo" in str(value) or "IA" in str(value) for row in reveal_values for value in row.values())

    metric_values = {metric.label: metric.value for metric in at.metric}
    assert any("1" in value for label, value in metric_values.items() if "catálogo" in label.lower())
    assert any("1" in value for label, value in metric_values.items() if label.lower().startswith("da ia"))


def test_save_button_disabled_until_process_code_and_experience_are_set(_log_dir, _candidates_path):
    at = _choose_one_catalog_and_one_ai_item(_generate(_run()))

    save_button = next(btn for btn in at.button if btn.key == "btn_save_evaluation")
    assert save_button.disabled

    at = _fill_mandatory_final_form(at)
    save_button = next(btn for btn in at.button if btn.key == "btn_save_evaluation")
    assert not save_button.disabled


def test_saving_writes_schema_v2_log_and_ai_candidate(_log_dir, _candidates_path):
    at = _choose_one_catalog_and_one_ai_item(_generate(_run()))
    at = _fill_mandatory_final_form(at)

    at.button(key="btn_save_evaluation").click()
    at.run()
    assert not at.exception, at.exception

    log_files = list(_log_dir.glob("*.json"))
    assert len(log_files) == 1
    data = json.loads(log_files[0].read_text(encoding="utf-8"))
    assert data["versao_schema"] == 2
    assert data["codigo_processo"] == "PROC-001"
    assert data["experiencia"] == "2_a_5"
    assert data["totais"]["tamanho_matriz_final"] == 2
    assert data["totais"]["escolhidos_catalogo"] == 1
    assert data["totais"]["escolhidos_ia"] == 1

    assert _candidates_path.exists()
    assert _candidates_path.read_text(encoding="utf-8").strip() != ""


def test_saving_again_without_changes_does_not_duplicate_the_log(_log_dir, _candidates_path):
    at = _choose_one_catalog_and_one_ai_item(_generate(_run()))
    at = _fill_mandatory_final_form(at)

    at.button(key="btn_save_evaluation").click()
    at.run()
    assert not at.exception, at.exception
    save_button = next(btn for btn in at.button if btn.key == "btn_save_evaluation")
    assert save_button.disabled
    assert len(list(_log_dir.glob("*.json"))) == 1

    at.text_area(key="likert_comentario").input("Mudei de ideia sobre o comentario")
    at.run()
    save_button = next(btn for btn in at.button if btn.key == "btn_save_evaluation")
    assert not save_button.disabled

    at.button(key="btn_save_evaluation").click()
    at.run()
    assert not at.exception, at.exception

    log_files = list(_log_dir.glob("*.json"))
    assert len(log_files) == 1
    data = json.loads(log_files[0].read_text(encoding="utf-8"))
    assert data["respostas_finais"]["comentario"] == "Mudei de ideia sobre o comentario"
