import csv
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "consolidar_avaliacao.py"
_spec = importlib.util.spec_from_file_location("consolidar_avaliacao", MODULE_PATH)
consolidar_avaliacao = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(consolidar_avaliacao)


def _sample_log(
    codigo: str,
    *,
    experiencia: str = "2_a_5",
    escolhidos_catalogo: int = 2,
    escolhidos_ia: int = 1,
    escolhidos_manual: int = 1,
    mostrados_ia: int = 3,
    riscos_manuais: int = 1,
    versao_schema: int = 2,
) -> dict:
    tamanho = escolhidos_catalogo + escolhidos_ia + escolhidos_manual

    def pct(count: int) -> float:
        return round(count / tamanho * 100, 1) if tamanho else 0.0

    return {
        "versao_schema": versao_schema,
        "codigo_processo": codigo,
        "nome_caso": "Caso exemplo",
        "experiencia": experiencia,
        "data_hora": "2026-01-01T00:00:00+00:00",
        "idioma": "pt",
        "provedor_ia": "anthropic",
        "modelo_ia": "claude-test",
        "versao_prompt": "v2",
        "hash_biblioteca": "hash123",
        "semente": 1,
        "sugestoes": [],
        "riscos_manuais": [{"id": f"MAN{i:03d}"} for i in range(1, riscos_manuais + 1)],
        "totais": {
            "mostrados_catalogo": 5,
            "mostrados_ia": mostrados_ia,
            "escolhidos_catalogo": escolhidos_catalogo,
            "escolhidos_ia": escolhidos_ia,
            "escolhidos_manual": escolhidos_manual,
            "tamanho_matriz_final": tamanho,
            "participacao_catalogo_pct": pct(escolhidos_catalogo),
            "participacao_ia_pct": pct(escolhidos_ia),
            "participacao_manual_pct": pct(escolhidos_manual),
            "adocao_ia_pct": round(escolhidos_ia / mostrados_ia * 100, 1) if mostrados_ia else 0.0,
            "editados_catalogo": 1,
            "editados_ia": 0,
            "por_categoria": [
                {
                    "categoria": "planejamento",
                    "escolhidos_catalogo": escolhidos_catalogo,
                    "escolhidos_ia": escolhidos_ia,
                    "escolhidos_manual": escolhidos_manual,
                }
            ],
        },
        "falhas_ia": [],
        "respostas_finais": {"utilidade": 4, "confianca": 5, "esforco": 2, "comentario": "ok"},
    }


def test_process_summary_row_reads_totals_and_answers():
    log = _sample_log("P1")

    row = consolidar_avaliacao.process_summary_row(log)

    assert row["codigo_processo"] == "P1"
    assert row["nome_caso"] == "Caso exemplo"
    assert row["experiencia"] == "2_a_5"
    assert row["tamanho_matriz_final"] == 4
    assert row["escolhidos_catalogo"] == 2
    assert row["escolhidos_ia"] == 1
    assert row["escolhidos_manual"] == 1
    assert row["riscos_manuais"] == 1
    assert row["utilidade"] == 4
    assert row["comentario"] == "ok"


def test_category_summary_rows_aggregate_across_processes():
    logs = [
        _sample_log("P1", escolhidos_catalogo=2, escolhidos_ia=1, escolhidos_manual=0),
        _sample_log("P2", escolhidos_catalogo=1, escolhidos_ia=2, escolhidos_manual=1),
    ]

    rows = consolidar_avaliacao.category_summary_rows(logs)

    assert len(rows) == 1
    row = rows[0]
    assert row["categoria"] == "planejamento"
    assert row["escolhidos_catalogo"] == 3
    assert row["escolhidos_ia"] == 3
    assert row["escolhidos_manual"] == 1
    assert row["total"] == 7


def test_experience_band_falls_back_to_unknown_for_blank_or_unexpected_value():
    assert consolidar_avaliacao.experience_band({"experiencia": ""}) == consolidar_avaliacao.EXPERIENCE_UNKNOWN
    assert consolidar_avaliacao.experience_band({"experiencia": "junior"}) == consolidar_avaliacao.EXPERIENCE_UNKNOWN
    assert consolidar_avaliacao.experience_band({"experiencia": "mais_5"}) == "mais_5"


def test_load_logs_skips_old_schema_with_warning(tmp_path, capsys):
    log_dir = tmp_path / "avaliacao"
    log_dir.mkdir()
    (log_dir / "novo.json").write_text(json.dumps(_sample_log("P1")), encoding="utf-8")
    (log_dir / "antigo.json").write_text(
        json.dumps({"codigo_processo": "OLD", "pares": []}), encoding="utf-8"
    )

    logs = consolidar_avaliacao.load_logs(log_dir)

    assert len(logs) == 1
    assert logs[0]["codigo_processo"] == "P1"
    captured = capsys.readouterr()
    assert "schema antigo" in captured.out
    assert "antigo.json" in captured.out


def test_consolidate_writes_csv_files(tmp_path):
    log_dir = tmp_path / "avaliacao"
    log_dir.mkdir()
    (log_dir / "p1.json").write_text(json.dumps(_sample_log("P1")), encoding="utf-8")
    (log_dir / "p2.json").write_text(json.dumps(_sample_log("P2")), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    process_rows, category_rows = consolidar_avaliacao.consolidate(log_dir, out_dir)

    assert len(process_rows) == 2
    assert len(category_rows) == 1
    with (out_dir / "resumo_processos.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    with (out_dir / "resumo_categorias.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_consolidate_with_empty_directory_writes_nothing(tmp_path):
    log_dir = tmp_path / "avaliacao"
    log_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    process_rows, category_rows = consolidar_avaliacao.consolidate(log_dir, out_dir)

    assert process_rows == []
    assert category_rows == []
    assert not (out_dir / "resumo_processos.csv").exists()


def test_print_report_shows_overall_and_per_experience_band(capsys):
    process_rows = [
        consolidar_avaliacao.process_summary_row(_sample_log("P1", experiencia="ate_2")),
        consolidar_avaliacao.process_summary_row(_sample_log("P2", experiencia="mais_5")),
        consolidar_avaliacao.process_summary_row(_sample_log("P3", experiencia="mais_5")),
    ]

    consolidar_avaliacao.print_report(process_rows)

    captured = capsys.readouterr()
    assert "Processos: 3" in captured.out
    assert "Geral: 3 processo(s)" in captured.out
    assert "Experiencia: Ate 2 anos: 1 processo(s)" in captured.out
    assert "Experiencia: Mais de 5 anos: 2 processo(s)" in captured.out
    assert "Participacao media da IA" in captured.out
    assert "Adocao media das sugestoes de IA" in captured.out
    # Nao e uma comparacao direta IA x catalogo: nao ha McNemar nem p-valor.
    assert "McNemar" not in captured.out
    assert "p-valor" not in captured.out


def test_main_runs_end_to_end_and_prints_report(tmp_path, capsys):
    log_dir = tmp_path / "avaliacao"
    log_dir.mkdir()
    (log_dir / "p1.json").write_text(json.dumps(_sample_log("P1")), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    consolidar_avaliacao.main(["--log-dir", str(log_dir), "--out-dir", str(out_dir)])

    captured = capsys.readouterr()
    assert "Processos: 1" in captured.out
    assert (out_dir / "resumo_processos.csv").exists()
    assert (out_dir / "resumo_categorias.csv").exists()


def test_main_with_only_old_schema_logs_reports_none_found(tmp_path, capsys):
    log_dir = tmp_path / "avaliacao"
    log_dir.mkdir()
    (log_dir / "antigo.json").write_text(
        json.dumps({"codigo_processo": "OLD", "pares": []}), encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    consolidar_avaliacao.main(["--log-dir", str(log_dir), "--out-dir", str(out_dir)])

    captured = capsys.readouterr()
    assert "schema antigo" in captured.out
    assert "Nenhum log de schema v2 encontrado" in captured.out
