import csv
import json

from src.risk_library import load_risks

from evaluation.cases.schema import case_from_dict
from evaluation.config import EmbeddingConfig, LLMConfig, RunConfig
from evaluation.engines.embedding_engine import HashingEmbeddingBackend
from evaluation.runner import run_evaluation, write_report


class FakeLLMClient:
    """Dublê de LLMClient: sempre devolve a mesma resposta, independente do
    prompt. Suficiente para os testes abaixo, que usam um unico caso por
    execucao -- o prompt real nao inclui o case_id (so campos de
    ContractContext), entao nao ha como distinguir casos pelo texto do prompt."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._response


def _case(case_id: str, gabarito_ids: list[str], used_to_build_base: bool = False):
    return case_from_dict(
        {
            "case_id": case_id,
            "context": {
                "objeto": "Aquisicao de solucao de software de seguranca da informacao",
                "tipo_contratacao": "software",
                "valor_estimado": 300000.0,
                "criticidade": "alta",
                "prazo": "12 meses",
                "modalidade": "pregao",
                "contexto": "seguranca, dados, acesso, lgpd",
            },
            "gabarito_ids": gabarito_ids,
            "used_to_build_base": used_to_build_base,
        }
    )


def test_run_evaluation_excludes_leaked_cases_and_produces_records(synthetic_risks, tmp_path, monkeypatch):
    risks_path = tmp_path / "riscos.csv"
    _write_risks_csv(risks_path, synthetic_risks)

    cases = [
        _case("eval-1", ["S001", "S003"]),
        _case("eval-2", ["S002"]),
        _case("leaked-1", ["S001"], used_to_build_base=True),
    ]
    config = RunConfig(
        engines=("rules", "bm25", "embedding"),
        embedding=EmbeddingConfig(backend="hashing"),
        reproducibility_repeats=2,
        bootstrap_resamples=200,
    )

    report = run_evaluation(
        risks_path=risks_path,
        config=config,
        include_real_cases=False,
        extra_cases=cases,
        embedding_backend=HashingEmbeddingBackend(),
    )

    assert {case.case_id for case in report.provenance.main_eval_cases} == {"eval-1", "eval-2"}
    assert {case.case_id for case in report.provenance.excluded_leaked_cases} == {"leaked-1"}

    evaluated_case_ids = {record.case_id for record in report.records}
    assert "leaked-1" not in evaluated_case_ids
    assert evaluated_case_ids == {"eval-1", "eval-2"}

    engines_present = {record.engine for record in report.records}
    assert engines_present == {"rules", "bm25", "embedding"}

    assert {aggregate.engine for aggregate in report.aggregates} == {"rules", "bm25", "embedding"}
    assert set(report.bootstrap_vs_baseline.keys()) == {"bm25", "embedding"}


def test_run_evaluation_includes_llm_when_enabled_and_tracks_invalid_ids(synthetic_risks, tmp_path):
    risks_path = tmp_path / "riscos.csv"
    _write_risks_csv(risks_path, synthetic_risks)

    cases = [_case("eval-1", ["S001"])]
    config = RunConfig(
        engines=("rules", "llm"),
        llm=LLMConfig(enabled=True),
        reproducibility_repeats=2,
    )
    client = FakeLLMClient('["S001", "FABRICATED-ID"]')

    report = run_evaluation(
        risks_path=risks_path,
        config=config,
        include_real_cases=False,
        extra_cases=cases,
        llm_client=client,
    )

    llm_record = next(record for record in report.records if record.engine == "llm")
    assert llm_record.invalid_id_rate == 0.5
    assert llm_record.predicted_top_10 == ("S001",)


def test_write_report_produces_json_csv_and_markdown(synthetic_risks, tmp_path):
    risks_path = tmp_path / "riscos.csv"
    _write_risks_csv(risks_path, synthetic_risks)

    cases = [_case("eval-1", ["S001", "S003"]), _case("eval-2", ["S002"])]
    config = RunConfig(
        engines=("rules", "bm25"),
        reproducibility_repeats=2,
        bootstrap_resamples=100,
    )

    report = run_evaluation(
        risks_path=risks_path,
        config=config,
        include_real_cases=False,
        extra_cases=cases,
    )

    output_dir = tmp_path / "report_out"
    write_report(report, output_dir)

    report_json = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report_json["provenance"]["main_eval_case_ids"] == ["eval-1", "eval-2"]
    assert len(report_json["records"]) == 4  # 2 casos x 2 motores

    with (output_dir / "per_case_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "rules" in summary
    assert "bm25" in summary


def _write_risks_csv(path, risks) -> None:
    from src.risk_library import FIELDNAMES

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for risk in risks:
            writer.writerow(
                {
                    "id": risk.id,
                    "titulo": risk.titulo,
                    "categoria": risk.categoria,
                    "tipo_contratacao": ";".join(risk.tipo_contratacao),
                    "palavras_chave": ";".join(risk.palavras_chave),
                    "causa": risk.causa,
                    "consequencia": risk.consequencia,
                    "probabilidade_padrao": risk.probabilidade_padrao,
                    "impacto_padrao": risk.impacto_padrao,
                    "acao_preventiva": risk.acao_preventiva,
                    "acao_contingencia": risk.acao_contingencia,
                    "responsavel_sugerido": risk.responsavel_sugerido,
                }
            )


def test_load_risks_roundtrip_matches_fixture(synthetic_risks, tmp_path):
    """Confere que o helper de escrita do CSV usado nos testes acima e fiel ao
    formato real lido por src.risk_library.load_risks (nenhuma alteracao no
    app: isto so documenta a dependencia entre os testes e o loader real)."""
    risks_path = tmp_path / "riscos.csv"
    _write_risks_csv(risks_path, synthetic_risks)

    reloaded = load_risks(risks_path)

    assert [risk.id for risk in reloaded] == [risk.id for risk in synthetic_risks]
