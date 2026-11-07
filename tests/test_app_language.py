"""Troca de idioma no app: nada do que o usuario preencheu pode se perder."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def _prob_boxes(at: AppTest):
    """Valores de probabilidade da revisao, pela chave estavel (sem o sufixo de idioma)."""
    return {box.key.split("@")[0]: box.value for box in at.selectbox if box.key and box.key.startswith("prob_")}


def _box(at: AppTest, key: str):
    return next(box for box in at.selectbox if box.key and box.key.split("@")[0] == key)


def _inclusion_checkboxes(at: AppTest):
    return [checkbox for checkbox in at.checkbox if checkbox.key and checkbox.key.startswith("incl_")]


def _generate(at: AppTest) -> AppTest:
    """Clica em 'Gerar sugestoes' (so entao a IA/pareamento sao chamados)."""
    at.button(key="btn_generate_suggestions").click()
    at.run()
    assert not at.exception, at.exception
    return at


def test_suggestions_are_not_generated_on_initial_load():
    at = _run()

    assert at.session_state.get("mix_result") is None
    assert not [box for box in at.text_input if box.key and box.key.startswith("risco_")]
    info_texts = [info.value for info in at.info]
    assert any("Gerar sugestões" in text for text in info_texts)


def test_default_language_is_portuguese():
    at = _run()

    assert [tab.label for tab in at.tabs] == ["Sugestões", "Revisão humana", "Exportação"]


def test_query_param_opens_in_english():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["lang"] = "en"
    at.run()

    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Suggestions", "Human review", "Export"]


def test_switching_language_keeps_inputs_and_blind_review_edits():
    at = _run()
    at.text_area(key="ctx_objeto").input("Contratação de licenças de software com implantação")
    _box(at, "ctx_criticidade").select("alta")
    _box(at, "ctx_modalidade").select("pregao_poc")
    at.number_input(key="ctx_prazo").set_value(24)
    at.button(key="btn_generate_suggestions").click()
    at.run()

    assert not at.exception, at.exception
    first_prob = sorted(_prob_boxes(at))[0]
    _box(at, first_prob).select(5)
    at.run()
    _inclusion_checkboxes(at)[0].set_value(True)
    at.run()
    before = _prob_boxes(at)
    items_total_before = at.metric[0].value

    at.radio(key="lang").set_value("en")
    at.run()

    assert not at.exception, at.exception
    assert [tab.label for tab in at.tabs] == ["Suggestions", "Human review", "Export"]
    assert at.text_area(key="ctx_objeto").value == "Contratação de licenças de software com implantação"
    assert _box(at, "ctx_criticidade").key == "ctx_criticidade@en"
    assert _box(at, "ctx_criticidade").value == "alta"
    assert _box(at, "ctx_modalidade").value == "pregao_poc"
    assert at.number_input(key="ctx_prazo").value == 24
    assert _prob_boxes(at) == before
    assert at.metric[0].value == items_total_before
    assert _box(at, "ctx_criticidade").format_func("alta") == "High"

    at.radio(key="lang").set_value("pt")
    at.run()

    assert _box(at, "ctx_criticidade").value == "alta"
    assert _box(at, "ctx_criticidade").format_func("alta") == "Alta"
    assert _prob_boxes(at) == before
    assert at.metric[0].value == items_total_before


def test_library_texts_follow_language_until_edited():
    at = _generate(_run())
    risk_inputs = sorted(box.key for box in at.text_input if box.key and box.key.startswith("risco_"))
    edited_key, untouched_key = risk_inputs[0], risk_inputs[1]
    at.text_input(key=edited_key).input("Texto revisado pelo usuário")
    at.run()
    untouched_pt = at.text_input(key=untouched_key).value

    at.radio(key="lang").set_value("en")
    at.run()

    assert not at.exception, at.exception
    assert at.text_input(key=edited_key).value == "Texto revisado pelo usuário"
    assert at.text_input(key=untouched_key).value != untouched_pt

    at.radio(key="lang").set_value("pt")
    at.run()

    assert at.text_input(key=untouched_key).value == untouched_pt
    assert at.text_input(key=edited_key).value == "Texto revisado pelo usuário"


def test_blind_suggestions_use_neutral_slot_ids_not_catalog_ids():
    at = _generate(_run())

    mix_result = at.session_state["mix_result"]
    assert mix_result.categories
    for category in mix_result.categories:
        for item in category.items:
            assert item.slot_id.startswith("S")
            if item.origin == "ia":
                assert item.row.id == ""
            assert not item.row.justificativa

    risco_keys = [box.key for box in at.text_input if box.key and box.key.startswith("risco_")]
    assert risco_keys
    assert all(key.split("@")[0].split("risco_", 1)[1].startswith("S") for key in risco_keys)
