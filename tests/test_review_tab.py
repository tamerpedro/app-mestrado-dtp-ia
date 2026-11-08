"""Aba Revisao (Parte 2): so os riscos escolhidos, edicao preservada, sem pista de origem."""

import re
from pathlib import Path

import pytest
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


_INCL_KEY_RE = re.compile(r"^incl_(S\d+)_(\d+)$")


def _inclusion_checkboxes(at: AppTest):
    return [checkbox for checkbox in at.checkbox if checkbox.key and checkbox.key.startswith("incl_")]


def _generation(at: AppTest) -> int:
    return at.session_state["mix_result"].seed


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
    generation = _generation(at)

    chosen_slot = next(_INCL_KEY_RE.match(cb.key).group(1) for cb in _inclusion_checkboxes(at) if cb.value)
    review_keys = _review_risco_keys(at)

    assert f"risco_{chosen_slot}_{generation}" in review_keys
    unchosen_slots = _all_slot_ids(at) - {chosen_slot}
    for slot in unchosen_slots:
        assert f"risco_{slot}_{generation}" not in review_keys


def test_empty_review_shows_message_when_nothing_chosen():
    at = _generate(_run())

    info_texts = [info.value for info in at.info]
    assert any("Sugestões" in text for text in info_texts)
    assert not _review_risco_keys(at)


def test_edits_are_preserved_when_unchecking_and_rechecking():
    at = _choose_items(_generate(_run()), count=1)
    generation = _generation(at)
    slot = next(_INCL_KEY_RE.match(cb.key).group(1) for cb in _inclusion_checkboxes(at) if cb.value)
    risco_key = f"risco_{slot}_{generation}"

    at.text_input(key=risco_key).input("Texto editado na revisão")
    at.run()
    assert at.text_input(key=risco_key).value == "Texto editado na revisão"

    at.checkbox(key=f"incl_{slot}_{generation}").set_value(False)
    at.run()
    assert not at.exception, at.exception
    assert risco_key not in _review_risco_keys(at)

    at.checkbox(key=f"incl_{slot}_{generation}").set_value(True)
    at.run()
    assert not at.exception, at.exception
    assert at.text_input(key=risco_key).value == "Texto editado na revisão"


def test_edit_is_measured_only_after_a_real_change_and_not_on_language_switch():
    at = _choose_items(_generate(_run()), count=1)
    generation = _generation(at)
    slot = next(_INCL_KEY_RE.match(cb.key).group(1) for cb in _inclusion_checkboxes(at) if cb.value)

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

    at.text_input(key=f"risco_{slot}_{generation}").input("Risco totalmente reescrito pelo usuário")
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


def test_manual_risk_appears_in_review_right_after_being_added():
    """Bug real reportado pelo usuario: ao adicionar um risco manual, ele so
    aparecia na Revisao (e na Exportacao) depois que outro risco manual era
    adicionado em seguida - ou seja, sempre um ciclo atrasado. A aba de
    Revisao le ``manual_rows`` antes do formulario (que fica mais abaixo na
    tela) processar a submissao, entao sem forcar um rerun o item novo nao
    aparecia na mesma interacao em que foi adicionado."""
    at = _run()

    risco_box = next(box for box in at.text_input if box.label == "Risco")
    risco_box.input("Risco manual de teste")
    submit = next(btn for btn in at.button if btn.label == "Adicionar risco")
    submit.click()
    at.run()

    assert not at.exception, at.exception
    review_box = next((box for box in at.text_input if box.key == "risco_MAN001"), None)
    assert review_box is not None, "o risco manual deveria aparecer na Revisao assim que adicionado"
    assert review_box.value == "Risco manual de teste"


def test_review_tab_never_reveals_catalog_or_ai_origin():
    at = _choose_items(_generate(_run()), count=3)

    review_keys = _review_risco_keys(at)
    assert review_keys
    # Chaves da Revisao sao sempre slot_id (S01, S02...) + geracao do sorteio, nunca o ID real do catalogo (R0xx).
    assert all(re.match(r"^risco_S\d{2,}_\d+$", key) for key in review_keys)
    assert not any(re.match(r"^risco_R\d{3}", key) for key in review_keys)

    review_decisions = at.session_state["review_decisions"]
    assert review_decisions
    assert all(re.match(r"^S\d{2,}$", slot) for slot in review_decisions)


def _row_for_slot(at: AppTest, slot: str):
    mix_result = at.session_state["mix_result"]
    for category in mix_result.categories:
        for item in category.items:
            if item.slot_id == slot:
                return item.row
    raise AssertionError(slot)


def test_regenerating_suggestions_does_not_leak_stale_state_into_a_reused_slot_id():
    """Bug real reportado pelo usuario: o mesmo slot_id (ex.: S32) pode calhar
    de apontar para um risco diferente a cada novo clique em "Gerar
    sugestoes" (o sorteio e aleatorio). Antes da correcao, o Streamlit
    reaproveitava o estado do widget (texto editado e caixa "incluir") de
    quem ocupou aquele slot_id na rodada anterior - deixando Risco/Causa em
    branco (ou com o texto do risco antigo) e a caixa ja marcada, sem o
    usuario ter feito nada com o risco novo."""
    at = _choose_items(_generate(_run()), count=1)
    generation = _generation(at)
    slot = next(_INCL_KEY_RE.match(cb.key).group(1) for cb in _inclusion_checkboxes(at) if cb.value)
    risco_key = f"risco_{slot}_{generation}"

    at.text_input(key=risco_key).input("")
    at.run()
    assert at.text_input(key=risco_key).value == ""

    original_risco = _row_for_slot(at, slot).risco
    for _ in range(20):
        at = _generate(at)
        new_row = _row_for_slot(at, slot)
        if new_row.risco != original_risco:
            break
    else:
        pytest.skip("sorteio nao produziu um risco diferente para o mesmo slot em 20 tentativas")

    new_generation = _generation(at)
    assert new_generation != generation

    new_checkbox = next(cb for cb in _inclusion_checkboxes(at) if cb.key == f"incl_{slot}_{new_generation}")
    assert new_checkbox.value is False, "caixa nao deveria vir pre-marcada por causa da rodada anterior"

    new_checkbox.set_value(True)
    at.run()
    new_risco_key = f"risco_{slot}_{new_generation}"
    assert at.text_input(key=new_risco_key).value == new_row.risco
