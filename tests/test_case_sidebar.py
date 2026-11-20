"""Com um caso selecionado, a barra lateral nao mostra os campos da contratacao."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CASE = str(Path("data/casos/exemplo.json"))
CONTEXT_KEYS = ["ctx_objeto", "ctx_tipo", "ctx_valor", "ctx_criticidade", "ctx_prazo", "ctx_modalidade", "ctx_contexto"]


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def _sidebar_keys(at: AppTest) -> set[str]:
    widgets = [
        *at.sidebar.text_area,
        *at.sidebar.text_input,
        *at.sidebar.number_input,
        *at.sidebar.selectbox,
    ]
    return {(widget.key or "").split("@")[0].split("_text_")[0] for widget in widgets}


def _case_box(at: AppTest):
    return next(box for box in at.sidebar.selectbox if (box.key or "").startswith("selected_case"))


def test_context_fields_appear_without_a_case():
    at = _run()

    assert set(CONTEXT_KEYS) <= _sidebar_keys(at)


def test_selecting_a_case_hides_context_fields_and_keeps_generate_button():
    at = _run()

    _case_box(at).select(CASE)
    at.run()

    assert not at.exception, at.exception
    assert not set(CONTEXT_KEYS) & _sidebar_keys(at)
    assert at.button(key="btn_generate_suggestions") is not None


def test_generating_with_a_case_works_without_the_fields():
    at = _run()
    at.checkbox(key="consentimento").set_value(True)
    _case_box(at).select(CASE)
    at.run()

    at.button(key="btn_generate_suggestions").click()
    at.run()

    assert not at.exception, at.exception


def test_returning_to_manual_mode_shows_the_fields_again():
    at = _run()
    _case_box(at).select(CASE)
    at.run()

    _case_box(at).select("__nenhum__")
    at.run()

    assert not at.exception, at.exception
    assert set(CONTEXT_KEYS) <= _sidebar_keys(at)
