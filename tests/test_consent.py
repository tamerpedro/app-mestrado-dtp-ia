"""Aviso de pesquisa e aceite obrigatorio (checkbox) na barra lateral."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_notice_and_checkbox_appear_at_the_top_of_the_sidebar_unchecked_by_default():
    at = _run()

    info_texts = [info.value for info in at.sidebar.info]
    assert any("Pesquisa acadêmica anônima" in text for text in info_texts)

    checkbox = at.sidebar.checkbox(key="consentimento")
    assert "participar voluntariamente da pesquisa" in checkbox.label
    assert checkbox.value is False


def test_generate_button_is_disabled_without_consent():
    at = _run()

    assert at.button(key="btn_generate_suggestions").disabled

    captions = [caption.value for caption in at.sidebar.caption]
    assert any("Aceite a participação na pesquisa" in text for text in captions)


def test_generate_button_is_enabled_after_accepting_consent():
    at = _run()

    at.checkbox(key="consentimento").set_value(True)
    at.run()

    assert not at.button(key="btn_generate_suggestions").disabled


def test_unchecking_consent_does_not_clear_filled_fields():
    at = _run()
    at.checkbox(key="consentimento").set_value(True)
    at.run()
    at.text_area(key="ctx_objeto").input("Contratação de licenças de software")
    at.run()

    at.checkbox(key="consentimento").set_value(False)
    at.run()

    assert not at.exception, at.exception
    assert at.text_area(key="ctx_objeto").value == "Contratação de licenças de software"
