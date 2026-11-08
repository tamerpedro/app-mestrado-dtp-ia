import json

from src.eval_log import (
    ORIGIN_AI,
    ORIGIN_CATALOG,
    EvaluationLog,
    ManualRiskLogEntry,
    SuggestionLogEntry,
    compute_totals,
    edit_ratio,
    save_evaluation_log,
)


def test_edit_ratio_is_zero_for_identical_text():
    assert edit_ratio("mesmo texto", "mesmo texto") == 0.0


def test_edit_ratio_is_zero_when_both_empty():
    assert edit_ratio("", "") == 0.0


def test_edit_ratio_is_high_for_completely_different_text():
    assert edit_ratio("abcdefgh", "zzzzzzzz") > 0.9


def test_edit_ratio_is_partial_for_small_change():
    ratio = edit_ratio("Risco de falha de integracao", "Risco de falha de integracao grave")
    assert 0.0 < ratio < 0.5


class _Row:
    """Dublê leve de MatrixRow: compute_totals so usa .id e .categoria."""

    def __init__(self, id_: str, categoria: str):
        self.id = id_
        self.categoria = categoria


def test_compute_totals_counts_mostrados_escolhidos_and_shares():
    items_by_slot = {
        "S01": {"origin": ORIGIN_CATALOG, "categoria": "planejamento", "incluido": True},
        "S02": {"origin": ORIGIN_AI, "categoria": "planejamento", "incluido": True},
        "S03": {"origin": ORIGIN_AI, "categoria": "planejamento", "incluido": False},
        "S04": {"origin": ORIGIN_CATALOG, "categoria": "selecao", "incluido": False},
    }
    selected_rows = [_Row("S01", "planejamento"), _Row("S02", "planejamento"), _Row("MAN001", "selecao")]
    slot_origin = {slot: info["origin"] for slot, info in items_by_slot.items()}

    totals = compute_totals(items_by_slot, review_decisions={}, selected_rows=selected_rows, slot_origin=slot_origin)

    assert totals.mostrados_catalogo == 2
    assert totals.mostrados_ia == 2
    assert totals.escolhidos_catalogo == 1
    assert totals.escolhidos_ia == 1
    assert totals.escolhidos_manual == 1
    assert totals.tamanho_matriz_final == 3
    assert totals.participacao_catalogo_pct == round(1 / 3 * 100, 1)
    assert totals.participacao_ia_pct == round(1 / 3 * 100, 1)
    assert totals.participacao_manual_pct == round(1 / 3 * 100, 1)


def test_compute_totals_ai_adoption_rate():
    items_by_slot = {
        "S01": {"origin": ORIGIN_AI, "categoria": "planejamento", "incluido": True},
        "S02": {"origin": ORIGIN_AI, "categoria": "planejamento", "incluido": False},
        "S03": {"origin": ORIGIN_AI, "categoria": "planejamento", "incluido": False},
        "S04": {"origin": ORIGIN_AI, "categoria": "planejamento", "incluido": False},
    }
    selected_rows = [_Row("S01", "planejamento")]
    slot_origin = {slot: info["origin"] for slot, info in items_by_slot.items()}

    totals = compute_totals(items_by_slot, {}, selected_rows, slot_origin)

    assert totals.adocao_ia_pct == 25.0


def test_compute_totals_ai_adoption_is_zero_when_nothing_shown():
    totals = compute_totals({}, {}, [], {})
    assert totals.adocao_ia_pct == 0.0
    assert totals.tamanho_matriz_final == 0
    assert totals.participacao_catalogo_pct == 0.0


def test_compute_totals_edited_counts_only_chosen_and_split_by_origin():
    items_by_slot = {
        "S01": {"origin": ORIGIN_CATALOG, "categoria": "planejamento", "incluido": True},
        "S02": {"origin": ORIGIN_AI, "categoria": "planejamento", "incluido": True},
    }
    review_decisions = {
        "S01": {"editado": True, "proporcao_edicao": 0.4},
        "S02": {"editado": False, "proporcao_edicao": 0.0},
    }
    selected_rows = [_Row("S01", "planejamento"), _Row("S02", "planejamento")]
    slot_origin = {slot: info["origin"] for slot, info in items_by_slot.items()}

    totals = compute_totals(items_by_slot, review_decisions, selected_rows, slot_origin)

    assert totals.editados_catalogo == 1
    assert totals.editados_ia == 0


def test_compute_totals_breaks_down_by_category():
    items_by_slot = {
        "S01": {"origin": ORIGIN_CATALOG, "categoria": "planejamento", "incluido": True},
        "S02": {"origin": ORIGIN_AI, "categoria": "selecao", "incluido": True},
    }
    selected_rows = [_Row("S01", "planejamento"), _Row("S02", "selecao"), _Row("MAN001", "selecao")]
    slot_origin = {slot: info["origin"] for slot, info in items_by_slot.items()}

    totals = compute_totals(items_by_slot, {}, selected_rows, slot_origin)

    by_category = {cat.categoria: cat for cat in totals.por_categoria}
    assert by_category["planejamento"].escolhidos_catalogo == 1
    assert by_category["selecao"].escolhidos_ia == 1
    assert by_category["selecao"].escolhidos_manual == 1


def _sample_log(codigo_processo: str = "PROC-001") -> EvaluationLog:
    log = EvaluationLog(
        codigo_processo=codigo_processo,
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=42,
        nome_caso="Caso de teste",
        experiencia="2_a_5",
    )
    log.sugestoes.append(
        SuggestionLogEntry(
            slot_id="S01",
            categoria="planejamento",
            origem="catalogo",
            id_catalogo="R001",
            posicao=1,
            escolhido=True,
            texto_mostrado={"risco": "Risco original"},
        )
    )
    log.sugestoes.append(
        SuggestionLogEntry(
            slot_id="S02",
            categoria="planejamento",
            origem="ia",
            id_catalogo=None,
            posicao=2,
            escolhido=False,
            texto_mostrado={"risco": "Risco de IA"},
        )
    )
    log.riscos_manuais.append(
        ManualRiskLogEntry(id="MAN001", categoria="selecao", escolhido=True, texto={"risco": "Risco manual"})
    )
    log.respostas_finais = {"utilidade": 4, "confianca": 5, "esforco": 2, "comentario": ""}
    return log


def test_save_evaluation_log_writes_expected_schema(tmp_path):
    log = _sample_log()

    path = save_evaluation_log(log, directory=tmp_path)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["versao_schema"] == 2
    assert data["nome_caso"] == "Caso de teste"
    assert data["experiencia"] == "2_a_5"
    for key in [
        "codigo_processo",
        "data_hora",
        "idioma",
        "provedor_ia",
        "modelo_ia",
        "versao_prompt",
        "hash_biblioteca",
        "semente",
        "sugestoes",
        "riscos_manuais",
        "totais",
        "falhas_ia",
        "respostas_finais",
    ]:
        assert key in data
    assert data["sugestoes"][0]["id_catalogo"] == "R001"
    assert data["sugestoes"][1]["id_catalogo"] is None
    assert data["riscos_manuais"][0]["id"] == "MAN001"


def test_save_evaluation_log_sanitizes_process_code_for_filename(tmp_path):
    log = _sample_log(codigo_processo="../../etc/passwd 2024/01")

    path = save_evaluation_log(log, directory=tmp_path)

    assert path.parent == tmp_path
    assert ".." not in path.name
    assert "/" not in path.name


def test_save_evaluation_log_creates_directory(tmp_path):
    nested = tmp_path / "avaliacao"
    log = _sample_log()

    path = save_evaluation_log(log, directory=nested)

    assert path.exists()
    assert nested.exists()


def test_save_evaluation_log_with_session_id_overwrites_same_file(tmp_path):
    log = _sample_log()

    first_path = save_evaluation_log(log, directory=tmp_path, session_id="sess123")
    log.respostas_finais["comentario"] = "mudei de ideia"
    second_path = save_evaluation_log(log, directory=tmp_path, session_id="sess123")

    assert first_path == second_path
    assert len(list(tmp_path.glob("*.json"))) == 1
    data = json.loads(second_path.read_text(encoding="utf-8"))
    assert data["respostas_finais"]["comentario"] == "mudei de ideia"


def test_save_evaluation_log_different_sessions_create_different_files(tmp_path):
    log = _sample_log()

    path_a = save_evaluation_log(log, directory=tmp_path, session_id="sess-a")
    path_b = save_evaluation_log(log, directory=tmp_path, session_id="sess-b")

    assert path_a != path_b
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_save_evaluation_log_without_session_id_creates_new_file_each_time(tmp_path):
    log = _sample_log()

    path_a = save_evaluation_log(log, directory=tmp_path)
    path_b = save_evaluation_log(log, directory=tmp_path)

    assert path_a != path_b
