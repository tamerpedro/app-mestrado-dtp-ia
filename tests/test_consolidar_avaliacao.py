import csv
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "consolidar_avaliacao.py"
_spec = importlib.util.spec_from_file_location("consolidar_avaliacao", MODULE_PATH)
consolidar_avaliacao = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(consolidar_avaliacao)


def _sample_log(codigo: str, outcomes: list[str]) -> dict:
    pares = [
        {"par": index, "categoria": f"cat{index}", "desfecho": desfecho}
        for index, desfecho in enumerate(outcomes, start=1)
    ]
    sugestoes = []
    for index, desfecho in enumerate(outcomes, start=1):
        if desfecho in ("so_biblioteca", "ambos"):
            sugestoes.append(
                {"par": index, "categoria": f"cat{index}", "origem": "biblioteca", "posicao": 1,
                 "slot_id": f"S{index:02d}", "escolhido": True, "editado": False, "proporcao_edicao": 0.0,
                 "motivo_descarte": None}
            )
        if desfecho in ("so_ia", "ambos"):
            sugestoes.append(
                {"par": index, "categoria": f"cat{index}", "origem": "ia", "posicao": 2,
                 "slot_id": f"S{index:02d}b", "escolhido": True, "editado": True, "proporcao_edicao": 0.4,
                 "motivo_descarte": None}
            )
    return {
        "codigo_processo": codigo,
        "data_hora": "2026-01-01T00:00:00+00:00",
        "idioma": "pt",
        "provedor_ia": "anthropic",
        "modelo_ia": "claude-test",
        "versao_prompt": "v1",
        "hash_biblioteca": "hash123",
        "semente": 1,
        "sugestoes": sugestoes,
        "pares": pares,
        "quantidade_riscos_manuais": 1,
        "falhas_ia": [],
        "respostas_finais": {"utilidade": 4, "confianca": 5, "esforco": 2, "comentario": "ok"},
    }


def test_mcnemar_exact_pvalue_is_one_when_balanced():
    assert consolidar_avaliacao.mcnemar_exact_pvalue(3, 3) == 1.0


def test_mcnemar_exact_pvalue_is_one_when_no_discordant_pairs():
    assert consolidar_avaliacao.mcnemar_exact_pvalue(0, 0) == 1.0


def test_mcnemar_exact_pvalue_matches_hand_computed_value():
    # n=6, k=1: (C(6,0)+C(6,1))/2^6 * 2 = (1+6)/64*2 = 0.21875
    assert round(consolidar_avaliacao.mcnemar_exact_pvalue(1, 5), 5) == 0.21875


def test_mcnemar_exact_pvalue_is_symmetric():
    assert consolidar_avaliacao.mcnemar_exact_pvalue(2, 8) == consolidar_avaliacao.mcnemar_exact_pvalue(8, 2)


def test_process_summary_row_aggregates_outcomes_and_answers():
    log = _sample_log("P1", ["so_biblioteca", "so_ia", "ambos", "nenhum", "sem_par"])

    row = consolidar_avaliacao.process_summary_row(log)

    assert row["codigo_processo"] == "P1"
    assert row["total_pares"] == 5
    assert row["so_biblioteca"] == 1
    assert row["so_ia"] == 1
    assert row["ambos"] == 1
    assert row["nenhum"] == 1
    assert row["sem_par"] == 1
    assert row["escolhidos_biblioteca"] == 2  # so_biblioteca + ambos
    assert row["escolhidos_ia"] == 2  # so_ia + ambos
    assert row["riscos_manuais"] == 1
    assert row["editados"] == 2  # as sugestoes de IA de so_ia e ambos
    assert row["utilidade"] == 4
    assert row["comentario"] == "ok"


def test_pair_summary_rows_carries_process_code():
    log = _sample_log("P2", ["so_biblioteca", "so_ia"])

    rows = consolidar_avaliacao.pair_summary_rows(log)

    assert len(rows) == 2
    assert all(row["codigo_processo"] == "P2" for row in rows)
    assert rows[0]["desfecho"] == "so_biblioteca"


def test_consolidate_writes_csv_files(tmp_path):
    log_dir = tmp_path / "avaliacao"
    log_dir.mkdir()
    (log_dir / "p1.json").write_text(json.dumps(_sample_log("P1", ["so_biblioteca", "so_ia"])), encoding="utf-8")
    (log_dir / "p2.json").write_text(json.dumps(_sample_log("P2", ["ambos"])), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    process_rows, pair_rows = consolidar_avaliacao.consolidate(log_dir, out_dir)

    assert len(process_rows) == 2
    assert len(pair_rows) == 3
    with (out_dir / "resumo_processos.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    with (out_dir / "resumo_pares.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_consolidate_with_empty_directory_writes_nothing(tmp_path):
    log_dir = tmp_path / "avaliacao"
    log_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    process_rows, pair_rows = consolidar_avaliacao.consolidate(log_dir, out_dir)

    assert process_rows == []
    assert pair_rows == []
    assert not (out_dir / "resumo_processos.csv").exists()


def test_main_runs_end_to_end_and_prints_report(tmp_path, capsys):
    log_dir = tmp_path / "avaliacao"
    log_dir.mkdir()
    (log_dir / "p1.json").write_text(
        json.dumps(_sample_log("P1", ["so_biblioteca", "so_ia", "so_biblioteca"])), encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    consolidar_avaliacao.main(["--log-dir", str(log_dir), "--out-dir", str(out_dir)])

    captured = capsys.readouterr()
    assert "Processos: 1" in captured.out
    assert "McNemar" in captured.out
    assert (out_dir / "resumo_processos.csv").exists()
    assert (out_dir / "resumo_pares.csv").exists()
