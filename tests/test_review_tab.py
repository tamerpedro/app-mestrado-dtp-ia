"""Aba Revisao (Parte 2): so os riscos escolhidos, edicao preservada, sem pista de origem."""

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


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


def _choose_items(at: AppTest, count: int = 1) -> AppTest:
    for checkbox in _inclusion_checkboxes(at)[:count]:
        checkbox.set_value(True)
    at.run()
    assert not at.exception, at.exception
    return at


def _review_risco_keys(at: AppTest) -> set[str]:
    return {box.key for box in at.text_input if box.key and box.key.startswith("risco_")}


def _all_slot_ids(at: AppTest) -> set[str]:
    mix_result = at.session_state["mix_result"]
    return {item.slot_id for category in mix_result.categories for item in category.items}


def test_unchosen_items_do_not_appear_in_review():
    at = _choose_items(_generate(_run()), count=1)

    chosen_slot = next(cb.key.removeprefix("incl_") for cb in _inclusion_checkboxes(at) if cb.value)
    review_keys = _review_risco_keys(at)

    assert f"risco_{chosen_slot}" in review_keys
    unchosen_slots = _all_slot_ids(at) - {chosen_slot}
    for slot in unchosen_slots:
        assert f"risco_{slot}" not in review_keys


def test_empty_review_shows_message_when_nothing_chosen():
    at = _generate(_run())

    info_texts = [info.value for info in at.info]
    assert any("Sugestões" in text for text in info_texts)
    assert not _review_risco_keys(at)


def test_edits_are_preserved_when_unchecking_and_rechecking():
    at = _choose_items(_generate(_run()), count=1)
    slot = next(cb.key.removeprefix("incl_") for cb in _inclusion_checkboxes(at) if cb.value)
    risco_key = f"risco_{slot}"

    at.text_input(key=risco_key).input("Texto editado na revisão")
    at.run()
    assert at.text_input(key=risco_key).value == "Texto editado na revisão"

    at.checkbox(key=f"incl_{slot}").set_value(False)
    at.run()
    assert not at.exception, at.exception
    assert risco_key not in _review_risco_keys(at)

    at.checkbox(key=f"incl_{slot}").set_value(True)
    at.run()
    assert not at.exception, at.exception
    assert at.text_input(key=risco_key).value == "Texto editado na revisão"


def test_edit_is_measured_only_after_a_real_change_and_not_on_language_switch():
    at = _choose_items(_generate(_run()), count=1)
    slot = next(cb.key.removeprefix("incl_") for cb in _inclusion_checkboxes(at) if cb.value)

    decisions = at.session_state["review_decisions"]
    assert decisions[slot]["editado"] is False
    assert decisions[slot]["proporcao_edicao"] == 0.0

    at.radio(key="lang").set_value("en")
    at.run()
    assert not at.exception, at.exception
    decisions = at.session_state["review_decisions"]
    assert decisions[slot]["editado"] is False

    at.radio(key="lang").set_value("pt")
    at.run()

    at.text_input(key=f"risco_{slot}").input("Risco totalmente reescrito pelo usuário")
    at.run()
    assert not at.exception, at.exception
    decisions = at.session_state["review_decisions"]
    assert decisions[slot]["editado"] is True
    assert decisions[slot]["proporcao_edicao"] > 0.0


def test_manual_risk_form_has_no_id_field_and_generates_id_automatically():
    at = _run()

    assert not any(box.label == "ID" for box in at.text_input)

    risco_box = next(box for box in at.text_input if box.label == "Risco")
    risco_box.input("Risco manual de teste")
    submit = next(btn for btn in at.button if btn.label == "Adicionar risco")
    submit.click()
    at.run()

    assert not at.exception, at.exception
    manual_rows = at.session_state["manual_rows"]
    assert len(manual_rows) == 1
    assert manual_rows[0].id == "MAN001"
    assert manual_rows[0].risco == "Risco manual de teste"


def test_review_tab_never_reveals_catalog_or_ai_origin():
    at = _choose_items(_generate(_run()), count=3)

    review_keys = _review_risco_keys(at)
    assert review_keys
    # Chaves da Revisao sao sempre slot_id (S01, S02...), nunca o ID real do catalogo (R0xx).
    assert all(re.match(r"^risco_S\d{2,}$", key) for key in review_keys)
    assert not any(re.match(r"^risco_R\d{3}$", key) for key in review_keys)

    review_decisions = at.session_state["review_decisions"]
    assert review_decisions
    assert all(re.match(r"^S\d{2,}$", slot) for slot in review_decisions)
