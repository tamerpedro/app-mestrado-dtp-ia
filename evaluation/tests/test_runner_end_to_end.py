import csv
import json

from src.risk_library import load_risks

from evaluation.cases.schema import case_from_dict
from evaluation.config import EmbeddingConfig, LLMConfig, RunConfig
from evaluation.engines.embedding_engine import HashingEmbeddingBackend
from evaluation.runner import (
    CaseEngineRecord,
    aggregate_engine_records,
    evaluate_case_with_engine,
    run_evaluation,
    write_report,
)


class FakeLLMClient:
    """Dublê de LLMClient: sempre devolve a mesma resposta, independente do
    prompt. Suficiente para os testes abaixo, que usam um unico caso por
    execucao -- o prompt real nao inclui o case_id (so campos de
    ContractContext), entao nao ha como distinguir casos pelo texto do prompt."""

    def __init__(self, response: str, last_model_id: str | None = None) -> None:
        self._response = response
        self.last_model_id = last_model_id

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

    # rules_production (os cortes de producao do app) e calculado
    # automaticamente ao lado de "rules" (ranking completo).
    engines_present = {record.engine for record in report.records}
    assert engines_present == {"rules", "rules_production", "bm25", "embedding"}

    assert {aggregate.engine for aggregate in report.aggregates} == {
        "rules",
        "rules_production",
        "bm25",
        "embedding",
    }
    # rules_production e informativo: nunca entra no bootstrap pareado.
    assert set(report.bootstrap_vs_baseline.keys()) == {"bm25", "embedding"}

    records_by_engine_and_case = {(r.engine, r.case_id): r for r in report.records}
    for case_id in ("eval-1", "eval-2"):
        full_ranking = records_by_engine_and_case[("rules", case_id)].predicted_top_10
        production = records_by_engine_and_case[("rules_production", case_id)].predicted_top_10
        assert set(production) <= set(full_ranking)
        assert len(production) <= len(full_ranking)


def test_run_evaluation_includes_llm_when_enabled_and_tracks_invalid_ids(synthetic_risks, tmp_path):
    risks_path = tmp_path / "riscos.csv"
    _write_risks_csv(risks_path, synthetic_risks)

    cases = [_case("eval-1", ["S001"])]
    config = RunConfig(
        engines=("rules", "llm"),
        llm=LLMConfig(enabled=True, model="llama3.1:8b-instruct"),
        reproducibility_repeats=2,
    )
    client = FakeLLMClient('["S001", "FABRICATED-ID"]', last_model_id="llama3.1:8b-instruct-q4_0")

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
    # Identificador exato do modelo (pode diferir da tag configurada): por caso...
    assert llm_record.llm_reported_model_id == "llama3.1:8b-instruct-q4_0"
    # ... e agregado no relatorio.
    llm_aggregate = next(agg for agg in report.aggregates if agg.engine == "llm")
    assert llm_aggregate.distinct_llm_reported_model_ids == ("llama3.1:8b-instruct-q4_0",)

    # Resposta valida (mesmo com um ID fabricado misturado) nao e uma falha de resposta/parsing.
    assert llm_record.llm_response_parse_failed is False
    assert llm_aggregate.llm_response_parse_failure_count == 0

    # rules_production continua sendo calculado automaticamente ao lado de "rules".
    assert {record.engine for record in report.records} == {"rules", "rules_production", "llm"}


def test_llm_response_parse_failures_are_tracked_per_case_and_aggregated(synthetic_risks, tmp_path):
    """Resposta vazia ou JSON invalido do LLM nunca deve virar um ranking
    vazio silencioso -- precisa ficar marcada no registro do caso e contada
    no agregado do motor."""
    risks_path = tmp_path / "riscos.csv"
    _write_risks_csv(risks_path, synthetic_risks)

    cases = [_case("eval-1", ["S001"]), _case("eval-2", ["S002"])]
    config = RunConfig(
        engines=("rules", "llm"),
        llm=LLMConfig(enabled=True),
        reproducibility_repeats=2,
    )
    client = FakeLLMClient("isto nao e JSON valido")  # mesma resposta malformada para todas as chamadas

    report = run_evaluation(
        risks_path=risks_path,
        config=config,
        include_real_cases=False,
        extra_cases=cases,
        llm_client=client,
    )

    llm_records = [record for record in report.records if record.engine == "llm"]
    assert len(llm_records) == 2
    assert all(record.llm_response_parse_failed for record in llm_records)
    assert all(record.predicted_top_10 == () for record in llm_records)

    llm_aggregate = next(agg for agg in report.aggregates if agg.engine == "llm")
    assert llm_aggregate.llm_response_parse_failure_count == 2

    # motores que nao sao o LLM nunca tem falha de resposta registrada.
    rules_aggregate = next(agg for agg in report.aggregates if agg.engine == "rules")
    assert rules_aggregate.llm_response_parse_failure_count == 0


def test_evaluate_case_with_engine_flags_llm_parse_failure_on_a_single_case(synthetic_risks):
    case = _case("eval-1", ["S001"])
    known_ids = frozenset(risk.id for risk in synthetic_risks)
    config = RunConfig(llm=LLMConfig(enabled=True), reproducibility_repeats=2)
    client = FakeLLMClient("")  # resposta vazia

    record = evaluate_case_with_engine(case, "llm", synthetic_risks, known_ids, config, llm_client=client)

    assert record.llm_response_parse_failed is True
    assert record.predicted_top_10 == ()
    assert record.recall_at_10 == 0.0


def test_aggregate_engine_records_counts_parse_failures_across_cases():
    def _record(case_id: str, parse_failed: bool) -> CaseEngineRecord:
        return CaseEngineRecord(
            case_id=case_id,
            engine="llm",
            precision_at_5=0.0,
            precision_at_10=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            ndcg_at_10=0.0,
            mrr=0.0,
            invalid_id_rate=None,
            execution_time_s=0.001,
            reproducibility_exact_match_rate=1.0,
            reproducibility_mean_top_k_jaccard=1.0,
            predicted_top_10=(),
            llm_response_parse_failed=parse_failed,
        )

    records = [_record("c1", True), _record("c2", False), _record("c3", True)]

    aggregate = aggregate_engine_records("llm", records)

    assert aggregate.llm_response_parse_failure_count == 2
    assert aggregate.n_cases == 3


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
    assert report_json["primary_metric"] == "recall_at_10"
    assert "recall_at_10" not in report_json["exploratory_metrics"]
    # 2 casos x 3 "motores" (rules, rules_production e bm25 -- rules_production
    # e calculado automaticamente ao lado de "rules").
    assert len(report_json["records"]) == 6

    with (output_dir / "per_case_metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "rules" in summary
    assert "bm25" in summary
    assert "rules_production" in summary
    assert "principal" in summary  # rotula a metrica principal (recall_at_10) no texto


def test_bootstrap_lists_the_primary_metric_first(synthetic_risks, tmp_path):
    risks_path = tmp_path / "riscos.csv"
    _write_risks_csv(risks_path, synthetic_risks)

    cases = [_case("eval-1", ["S001", "S003"]), _case("eval-2", ["S002"])]
    config = RunConfig(engines=("rules", "bm25"), reproducibility_repeats=2, bootstrap_resamples=100)

    report = run_evaluation(
        risks_path=risks_path,
        config=config,
        include_real_cases=False,
        extra_cases=cases,
    )

    metric_order = list(report.bootstrap_vs_baseline["bm25"].keys())
    assert metric_order[0] == "recall_at_10" == config.primary_metric


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
