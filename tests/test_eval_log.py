import json

from src.eval_log import EvaluationLog, SuggestionLogEntry, edit_ratio, save_evaluation_log


def test_edit_ratio_is_zero_for_identical_text():
    assert edit_ratio("mesmo texto", "mesmo texto") == 0.0


def test_edit_ratio_is_zero_when_both_empty():
    assert edit_ratio("", "") == 0.0


def test_edit_ratio_is_high_for_completely_different_text():
    assert edit_ratio("abcdefgh", "zzzzzzzz") > 0.9


def test_edit_ratio_is_partial_for_small_change():
    ratio = edit_ratio("Risco de falha de integracao", "Risco de falha de integracao grave")
    assert 0.0 < ratio < 0.5


def _sample_log(codigo_processo: str = "PROC-001") -> EvaluationLog:
    log = EvaluationLog(
        codigo_processo=codigo_processo,
        idioma="pt",
        provedor_ia="anthropic",
        modelo_ia="claude-test",
        versao_prompt="v2",
        hash_biblioteca="abc123",
        semente=42,
    )
    log.sugestoes.append(
        SuggestionLogEntry(categoria="planejamento", origem="catalogo", posicao=1, slot_id="S01", escolhido=True)
    )
    log.sugestoes.append(
        SuggestionLogEntry(categoria="planejamento", origem="ia", posicao=2, slot_id="S02", escolhido=False)
    )
    log.quantidade_riscos_manuais = 1
    log.respostas_finais = {"utilidade": 4, "confianca": 5, "esforco": 2, "comentario": ""}
    return log


def test_save_evaluation_log_writes_expected_schema(tmp_path):
    log = _sample_log()

    path = save_evaluation_log(log, directory=tmp_path)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
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
        "quantidade_riscos_manuais",
        "respostas_finais",
    ]:
        assert key in data
    assert data["sugestoes"][0]["origem"] == "catalogo"
    assert data["sugestoes"][1]["origem"] == "ia"
    assert data["quantidade_riscos_manuais"] == 1


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
