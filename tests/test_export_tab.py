"""Aba Exportacao: resultado da avaliacao e exportacao do log (sem gravar em disco)."""

import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.ai_provider as ai_provider_module
import src.eval_log as eval_log_module
import src.risk_library as risk_library_module
from src.ai_provider import FakeProvider

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


@pytest.fixture(autouse=True)
def _fail_if_anything_is_saved_to_disk(monkeypatch):
    """O fluxo de exportacao nao deve gravar nada no servidor: qualquer
    chamada a save_evaluation_log ou save_matrix_row_to_library aqui e um
    regressao (a Parte 1 trocou "Salvar avaliacao" por um download)."""

    def _fail_save_log(*args, **kwargs):
        raise AssertionError("save_evaluation_log nao deveria ser chamado pela interface")

    def _fail_save_row(*args, **kwargs):
        raise AssertionError("save_matrix_row_to_library nao deveria ser chamado pela interface neste fluxo")

    monkeypatch.setattr(eval_log_module, "save_evaluation_log", _fail_save_log)
    monkeypatch.setattr(risk_library_module, "save_matrix_row_to_library", _fail_save_row)


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def _generate(at: AppTest) -> AppTest:
    at.button(key="btn_generate_suggestions").click()
    at.run()
    assert not at.exception, at.exception
    return at


_INCL_KEY_RE = re.compile(r"^incl_(S\d+)_\d+$")


def _inclusion_checkboxes(at: AppTest):
    return [checkbox for checkbox in at.checkbox if checkbox.key and checkbox.key.startswith("incl_")]


def _slot_of_inclusion_checkbox(key: str) -> str:
    match = _INCL_KEY_RE.match(key)
    assert match, key
    return match.group(1)


def _slot_origin(at: AppTest) -> dict[str, str]:
    mix_result = at.session_state["mix_result"]
    return {item.slot_id: item.origin for category in mix_result.categories for item in category.items}


def _choose_one_catalog_and_one_ai_item(at: AppTest) -> AppTest:
    origin = _slot_origin(at)
    checkboxes = _inclusion_checkboxes(at)
    catalog_box = next(cb for cb in checkboxes if origin.get(_slot_of_inclusion_checkbox(cb.key)) == "catalogo")
    ai_box = next(cb for cb in checkboxes if origin.get(_slot_of_inclusion_checkbox(cb.key)) == "ia")
    catalog_box.set_value(True)
    ai_box.set_value(True)
    at.run()
    assert not at.exception, at.exception
    return at


def _export_button(at: AppTest):
    return next(btn for btn in at.download_button if btn.key == "btn_export_log")


def test_export_button_disabled_without_process_code_or_experience():
    at = _choose_one_catalog_and_one_ai_item(_generate(_run()))

    assert _export_button(at).disabled


def test_export_button_disabled_with_process_code_but_no_experience():
    at = _choose_one_catalog_and_one_ai_item(_generate(_run()))

    at.text_input(key="codigo_processo").input("PROC-001")
    at.run()

    assert _export_button(at).disabled


def test_export_button_enabled_once_process_code_and_experience_are_set():
    at = _choose_one_catalog_and_one_ai_item(_generate(_run()))

    at.text_input(key="codigo_processo").input("PROC-001")
    at.run()
    at.radio(key="experiencia").set_value("2_a_5")
    at.run()
    assert not at.exception, at.exception

    assert not _export_button(at).disabled


def test_result_section_shows_composition_metrics_and_reveal_table():
    at = _choose_one_catalog_and_one_ai_item(_generate(_run()))

    reveal_values = [row for df in at.dataframe for row in df.value.to_dict("records")]
    assert any("Catálogo" in str(value) or "IA" in str(value) for row in reveal_values for value in row.values())

    metric_values = {metric.label: metric.value for metric in at.metric}
    assert any("1" in value for label, value in metric_values.items() if "catálogo" in label.lower())
    assert any("1" in value for label, value in metric_values.items() if label.lower().startswith("da ia"))


def test_nothing_is_written_to_disk_when_filling_and_rendering_the_final_form():
    at = _choose_one_catalog_and_one_ai_item(_generate(_run()))

    at.text_input(key="codigo_processo").input("PROC-001")
    at.run()
    at.radio(key="experiencia").set_value("2_a_5")
    at.run()
    at.text_area(key="likert_comentario").input("Comentario de teste")
    at.run()

    assert not at.exception, at.exception
    assert not _export_button(at).disabled
