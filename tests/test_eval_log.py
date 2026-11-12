import json

from src.eval_log import (
    ORIGIN_AI,
    ORIGIN_CATALOG,
    EvaluationLog,
    ManualRiskLogEntry,
    SuggestionLogEntry,
    build_evaluation_log,
    compute_totals,
    edit_ratio,
    log_consistency_errors,
    save_evaluation_log,
)
from src.models import ActionItem, MatrixRow


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


def _matrix_row(id_: str, categoria: str, risco: str) -> MatrixRow:
    return MatrixRow(
        id=id_,
        risco=risco,
        categoria=categoria,
        causa="Causa",
        consequencias=["Consequencia"],
        probabilidade=3,
        impacto=3,
        nivel="moderado",
        estrategia="mitigar",
        acoes_preventivas=[ActionItem("Preventiva")],
        acoes_contingencia=[ActionItem("Contingencia")],
    )


def _sample_items_by_slot() -> dict:
    return {
        "S01": {
            "row": MatrixRow(
                id="R001",
                risco="x",
                categoria="planejamento",
                causa="",
                consequencias=[],
                probabilidade=3,
                impacto=3,
                nivel="moderado",
                estrategia="mitigar",
                acoes_preventivas=[],
                acoes_contingencia=[],
            ),
            "origin": ORIGIN_CATALOG,
            "categoria": "planejamento",
            "posicao": 1,
            "incluido": True,
            "defaults": {
                "risco": "Risco original",
                "causa": "Causa original",
                "consequencias": ["Consequencia original"],
                "preventivas": [ActionItem("Preventiva original")],
                "contingencias": [ActionItem("Contingencia original")],
            },
        },
        "S02": {
            "row": MatrixRow(
                id="",
                risco="y",
                categoria="planejamento",
                causa="",
                consequencias=[],
                probabilidade=3,
                impacto=3,
                nivel="moderado",
                estrategia="mitigar",
                acoes_preventivas=[],
                acoes_contingencia=[],
            ),
            "origin": ORIGIN_AI,
            "categoria": "planejamento",
            "posicao": 2,
            "incluido": False,
            "defaults": {
                "risco": "Risco de IA",
                "causa": "Causa IA",
                "consequencias": ["Consequencia IA"],
                "preventivas": [ActionItem("Preventiva IA")],
                "contingencias": [ActionItem("Contingencia IA")],
            },
        },
    }


def test_build_evaluation_log_assembles_schema_v2_from_app_state():
    items_by_slot = _sample_items_by_slot()
    review_decisions = {"S01": {"editado": True, "proporcao_edicao": 0.3}}
    edited_rows = [_matrix_row("S01", "planejamento", "Risco editado")]
    manual_rows = [_matrix_row("MAN001", "selecao", "Risco manual")]
    selected_rows = [
        _matrix_row("S01", "planejamento", "Risco editado"),
        _matrix_row("MAN001", "selecao", "Risco manual"),
    ]

    log = build_evaluation_log(
        codigo_processo="PROC-1",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=7,
        nome_caso="Caso X",
        experiencia="2_a_5",
        items_by_slot=items_by_slot,
        review_decisions=review_decisions,
        edited_rows=edited_rows,
        manual_rows=manual_rows,
        selected_rows=selected_rows,
        falhas_ia=[],
        respostas_finais={"utilidade": 4, "confianca": 5, "esforco": 2, "comentario": "ok"},
    )

    assert log.versao_schema == 2
    assert log.codigo_processo == "PROC-1"
    assert log.nome_caso == "Caso X"
    assert log.experiencia == "2_a_5"
    assert len(log.sugestoes) == 2

    s01 = next(s for s in log.sugestoes if s.slot_id == "S01")
    assert s01.origem == ORIGIN_CATALOG
    assert s01.id_catalogo == "R001"
    assert s01.escolhido is True
    assert s01.editado is True
    assert s01.texto_mostrado["risco"] == "Risco original"
    assert s01.texto_final["risco"] == "Risco editado"
    # edited_rows traz S01 com o selecionado=True default do MatrixRow, e sem mudar de categoria.
    assert s01.incluido_matriz is True
    assert s01.categoria_final == "planejamento"
    assert s01.categoria_alterada is False

    s02 = next(s for s in log.sugestoes if s.slot_id == "S02")
    assert s02.origem == ORIGIN_AI
    assert s02.id_catalogo is None
    assert s02.escolhido is False
    assert s02.editado is False
    assert s02.texto_final is None
    # S02 nunca foi escolhido na aba Sugestoes, entao nem chega a Revisao.
    assert s02.incluido_matriz is False
    assert s02.categoria_final is None
    assert s02.categoria_alterada is False

    assert len(log.riscos_manuais) == 1
    assert log.riscos_manuais[0].id == "MAN001"
    assert log.riscos_manuais[0].escolhido is True
    assert log.riscos_manuais[0].texto["risco"] == "Risco manual"
    assert log.riscos_manuais[0].categoria == "selecao"
    assert log.riscos_manuais[0].categoria_final == "selecao"
    assert log.riscos_manuais[0].categoria_alterada is False

    assert log.totais.tamanho_matriz_final == 2
    assert log.totais.escolhidos_catalogo == 1
    assert log.totais.escolhidos_manual == 1
    assert log.respostas_finais == {"utilidade": 4, "confianca": 5, "esforco": 2, "comentario": "ok"}
    assert log_consistency_errors(log.to_dict()) == []


def test_build_evaluation_log_manual_risk_not_selected_is_not_escolhido():
    items_by_slot = _sample_items_by_slot()
    manual_rows = [_matrix_row("MAN001", "selecao", "Risco manual nao escolhido")]

    log = build_evaluation_log(
        codigo_processo="PROC-2",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=1,
        nome_caso=None,
        experiencia="ate_2",
        items_by_slot=items_by_slot,
        review_decisions={},
        edited_rows=[],
        manual_rows=manual_rows,
        selected_rows=[],
        falhas_ia=[],
        respostas_finais={},
    )

    assert log.riscos_manuais[0].escolhido is False
    assert log.totais.tamanho_matriz_final == 0
    assert log.riscos_manuais[0].categoria_final is None
    assert log.riscos_manuais[0].categoria_alterada is False
    assert log_consistency_errors(log.to_dict()) == []


def test_build_evaluation_log_tracks_unchecked_inclusion_and_category_changes():
    """Cobre os dois casos que o resumo/consolidacao precisa diferenciar:
    um item escolhido na aba Sugestoes mas desmarcado na Revisao (nao entra
    na matriz, categoria_final vira None), e um item que entrou na matriz
    mas com a "Categoria no mapa" alterada em relacao a categoria original -
    tanto para um item as cegas quanto para um risco manual."""
    items_by_slot = {
        "S01": {
            "row": MatrixRow(
                id="R001", risco="x", categoria="planejamento", causa="", consequencias=[],
                probabilidade=3, impacto=3, nivel="moderado", estrategia="mitigar",
                acoes_preventivas=[], acoes_contingencia=[],
            ),
            "origin": ORIGIN_CATALOG,
            "categoria": "planejamento",
            "posicao": 1,
            "incluido": True,
            "defaults": {
                "risco": "Risco original", "causa": "", "consequencias": [], "preventivas": [], "contingencias": [],
            },
        },
        "S02": {
            "row": MatrixRow(
                id="", risco="y", categoria="planejamento", causa="", consequencias=[],
                probabilidade=3, impacto=3, nivel="moderado", estrategia="mitigar",
                acoes_preventivas=[], acoes_contingencia=[],
            ),
            "origin": ORIGIN_AI,
            "categoria": "planejamento",
            "posicao": 2,
            "incluido": True,
            "defaults": {
                "risco": "Risco IA", "causa": "", "consequencias": [], "preventivas": [], "contingencias": [],
            },
        },
    }
    edited_rows = [
        _matrix_row("S01", "planejamento", "Risco editado"),  # desmarcado abaixo
        _matrix_row("S02", "selecao", "Risco IA editado"),  # categoria mudou de planejamento -> selecao
        _matrix_row("MAN001", "gestao", "Risco manual"),  # categoria mudou de selecao -> gestao
        _matrix_row("MAN002", "planejamento", "Risco manual 2"),  # desmarcado abaixo
    ]
    edited_rows[0].selecionado = False
    edited_rows[3].selecionado = False
    manual_rows = [
        _matrix_row("MAN001", "selecao", "Risco manual (original)"),
        _matrix_row("MAN002", "planejamento", "Risco manual 2 (original)"),
    ]
    selected_rows = [row for row in edited_rows if row.selecionado]

    log = build_evaluation_log(
        codigo_processo="PROC-4",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=1,
        nome_caso=None,
        experiencia="mais_5",
        items_by_slot=items_by_slot,
        review_decisions={},
        edited_rows=edited_rows,
        manual_rows=manual_rows,
        selected_rows=selected_rows,
        falhas_ia=[],
        respostas_finais={},
    )

    s01 = next(s for s in log.sugestoes if s.slot_id == "S01")
    assert s01.escolhido is True
    assert s01.incluido_matriz is False  # escolhido, mas desmarcado na Revisao
    assert s01.categoria_final is None
    assert s01.categoria_alterada is False

    s02 = next(s for s in log.sugestoes if s.slot_id == "S02")
    assert s02.incluido_matriz is True
    assert s02.categoria == "planejamento"
    assert s02.categoria_final == "selecao"
    assert s02.categoria_alterada is True

    man1 = next(m for m in log.riscos_manuais if m.id == "MAN001")
    assert man1.escolhido is True
    assert man1.categoria == "selecao"
    assert man1.categoria_final == "gestao"
    assert man1.categoria_alterada is True

    man2 = next(m for m in log.riscos_manuais if m.id == "MAN002")
    assert man2.escolhido is False
    assert man2.categoria_final is None
    assert man2.categoria_alterada is False

    assert log_consistency_errors(log.to_dict()) == []


def test_build_evaluation_log_tracks_avaliacao_changes_and_totals():
    """avaliacao_mostrada/avaliacao_final/avaliacao_alterada cobrem so
    probabilidade/impacto/estrategia - independente de categoria_alterada
    (que cobre a categoria) e de editado/proporcao_edicao (que cobrem so
    texto)."""
    items_by_slot = {
        "S01": {
            "row": MatrixRow(
                id="R001", risco="x", categoria="planejamento", causa="", consequencias=[],
                probabilidade=2, impacto=2, nivel="baixo", estrategia="mitigate",
                acoes_preventivas=[], acoes_contingencia=[],
            ),
            "origin": ORIGIN_CATALOG,
            "categoria": "planejamento",
            "posicao": 1,
            "incluido": True,
            "defaults": {
                "risco": "Risco original", "causa": "", "consequencias": [], "preventivas": [], "contingencias": [],
            },
        },
        "S02": {
            "row": MatrixRow(
                id="", risco="y", categoria="planejamento", causa="", consequencias=[],
                probabilidade=3, impacto=3, nivel="moderado", estrategia="mitigate",
                acoes_preventivas=[], acoes_contingencia=[],
            ),
            "origin": ORIGIN_AI,
            "categoria": "planejamento",
            "posicao": 2,
            "incluido": True,
            "defaults": {
                "risco": "Risco IA", "causa": "", "consequencias": [], "preventivas": [], "contingencias": [],
            },
        },
    }
    s01_edited = _matrix_row("S01", "planejamento", "Risco editado")
    s01_edited.probabilidade = 4
    s01_edited.impacto = 2  # nao muda, so probabilidade e estrategia mudam
    s01_edited.estrategia = "accept"
    s02_edited = _matrix_row("S02", "planejamento", "Risco IA editado")
    s02_edited.probabilidade = 3
    s02_edited.impacto = 3
    s02_edited.estrategia = "mitigate"
    edited_rows = [s01_edited, s02_edited]
    selected_rows = list(edited_rows)

    log = build_evaluation_log(
        codigo_processo="PROC-10",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=1,
        nome_caso=None,
        experiencia="mais_5",
        items_by_slot=items_by_slot,
        review_decisions={},
        edited_rows=edited_rows,
        manual_rows=[],
        selected_rows=selected_rows,
        falhas_ia=[],
        respostas_finais={},
    )

    s01 = next(s for s in log.sugestoes if s.slot_id == "S01")
    assert s01.avaliacao_mostrada == {"probabilidade": 2, "impacto": 2, "estrategia": "mitigate"}
    assert s01.avaliacao_final == {"probabilidade": 4, "impacto": 2, "estrategia": "accept"}
    assert s01.avaliacao_alterada is True
    # categoria nao mudou - os dois sinais sao independentes.
    assert s01.categoria_alterada is False

    s02 = next(s for s in log.sugestoes if s.slot_id == "S02")
    assert s02.avaliacao_mostrada == {"probabilidade": 3, "impacto": 3, "estrategia": "mitigate"}
    assert s02.avaliacao_final == {"probabilidade": 3, "impacto": 3, "estrategia": "mitigate"}
    assert s02.avaliacao_alterada is False

    assert log.totais.avaliacao_alterada_catalogo == 1
    assert log.totais.avaliacao_alterada_ia == 0
    assert log_consistency_errors(log.to_dict()) == []


def test_build_evaluation_log_avaliacao_final_is_none_when_not_included_in_matrix():
    items_by_slot = _sample_items_by_slot()

    log = build_evaluation_log(
        codigo_processo="PROC-11",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=1,
        nome_caso=None,
        experiencia="mais_5",
        items_by_slot=items_by_slot,
        review_decisions={},
        edited_rows=[],
        manual_rows=[],
        selected_rows=[],
        falhas_ia=[],
        respostas_finais={},
    )

    s02 = next(s for s in log.sugestoes if s.slot_id == "S02")
    assert s02.incluido_matriz is False
    assert s02.avaliacao_mostrada == {"probabilidade": 3, "impacto": 3, "estrategia": "mitigar"}
    assert s02.avaliacao_final is None
    assert s02.avaliacao_alterada is False


def test_build_evaluation_log_records_ai_failures():
    log = build_evaluation_log(
        codigo_processo="PROC-3",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=1,
        nome_caso=None,
        experiencia="mais_5",
        items_by_slot={},
        review_decisions={},
        edited_rows=[],
        manual_rows=[],
        selected_rows=[],
        falhas_ia=[{"categoria": "planejamento", "erro": "timeout"}],
        respostas_finais={},
    )

    assert log.falhas_ia == [{"categoria": "planejamento", "erro": "timeout"}]


def test_build_evaluation_log_defaults_consentimento_to_false():
    log = build_evaluation_log(
        codigo_processo="PROC-8",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=1,
        nome_caso=None,
        experiencia="mais_5",
        items_by_slot={},
        review_decisions={},
        edited_rows=[],
        manual_rows=[],
        selected_rows=[],
        falhas_ia=[],
        respostas_finais={},
    )

    assert log.consentimento is False
    assert log.versao_schema == 2


def test_build_evaluation_log_records_consentimento_when_accepted():
    log = build_evaluation_log(
        codigo_processo="PROC-9",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=1,
        nome_caso=None,
        experiencia="mais_5",
        items_by_slot={},
        review_decisions={},
        edited_rows=[],
        manual_rows=[],
        selected_rows=[],
        falhas_ia=[],
        respostas_finais={},
        consentimento=True,
    )

    assert log.consentimento is True
    assert log.versao_schema == 2
    assert log.to_dict()["consentimento"] is True


def test_log_consistency_errors_is_empty_for_a_freshly_built_log():
    log = build_evaluation_log(
        codigo_processo="PROC-5",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=7,
        nome_caso="Caso X",
        experiencia="2_a_5",
        items_by_slot=_sample_items_by_slot(),
        review_decisions={"S01": {"editado": True, "proporcao_edicao": 0.3}},
        edited_rows=[_matrix_row("S01", "planejamento", "Risco editado")],
        manual_rows=[_matrix_row("MAN001", "selecao", "Risco manual")],
        selected_rows=[
            _matrix_row("S01", "planejamento", "Risco editado"),
            _matrix_row("MAN001", "selecao", "Risco manual"),
        ],
        falhas_ia=[],
        respostas_finais={},
    )

    assert log_consistency_errors(log.to_dict()) == []


def test_log_consistency_errors_flags_a_tampered_total():
    log = build_evaluation_log(
        codigo_processo="PROC-6",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=7,
        nome_caso="Caso X",
        experiencia="2_a_5",
        items_by_slot=_sample_items_by_slot(),
        review_decisions={},
        edited_rows=[_matrix_row("S01", "planejamento", "Risco editado")],
        manual_rows=[],
        selected_rows=[_matrix_row("S01", "planejamento", "Risco editado")],
        falhas_ia=[],
        respostas_finais={},
    )
    data = log.to_dict()
    data["totais"]["escolhidos_catalogo"] = 99

    errors = log_consistency_errors(data)

    assert any("escolhidos_catalogo" in error for error in errors)


def test_log_consistency_errors_flags_a_tampered_category_breakdown():
    log = build_evaluation_log(
        codigo_processo="PROC-7",
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=7,
        nome_caso="Caso X",
        experiencia="2_a_5",
        items_by_slot=_sample_items_by_slot(),
        review_decisions={},
        edited_rows=[_matrix_row("S01", "planejamento", "Risco editado")],
        manual_rows=[],
        selected_rows=[_matrix_row("S01", "planejamento", "Risco editado")],
        falhas_ia=[],
        respostas_finais={},
    )
    data = log.to_dict()
    data["totais"]["por_categoria"][0]["escolhidos_catalogo"] = 0

    errors = log_consistency_errors(data)

    assert any("por_categoria" in error for error in errors)
