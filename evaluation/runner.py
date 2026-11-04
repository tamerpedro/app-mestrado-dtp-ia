"""Orquestra uma execucao completa da avaliacao: carrega riscos e casos,
roda cada motor habilitado sobre cada caso da avaliacao principal, calcula
metricas por caso, agrega por motor e compara o motor atual com os demais
via bootstrap pareado.

Uso tipico (ver tambem ``evaluation/README.md``):

    from evaluation.runner import run_evaluation, write_report
    from evaluation.config import RunConfig

    report = run_evaluation(risks_path="data/riscos_base.csv", config=RunConfig())
    write_report(report, output_dir="evaluation/reports/2026-01-01")
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.models import ContractContext, RiskItem
from src.risk_library import load_risks

from .cases.loader import load_real_cases, load_synthetic_cases
from .cases.provenance import ProvenanceReport, split_by_provenance
from .cases.schema import EvalCase
from .config import RunConfig
from .engines import bm25_engine, embedding_engine, llm_engine, rules_engine
from .engines.base import RankResult
from .engines.embedding_engine import EmbeddingBackend, build_default_backend
from .engines.llm_engine import LLMClient
from .metrics.bootstrap import PairedBootstrapResult, paired_bootstrap_ci
from .metrics.ranking import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k
from .metrics.reproducibility import ReproducibilityResult, measure_reproducibility
from .metrics.validity import invalid_id_rate
from .run_metadata import RunMetadata, collect_run_metadata

BASELINE_ENGINE = "rules"
# Variante informativa do motor de regras, com os cortes de producao
# (minimum_score e limite por categoria). E calculada automaticamente
# sempre que BASELINE_ENGINE roda, e aparece no relatorio ao lado do
# ranking completo -- mas fica fora do bootstrap pareado (nao e um motor
# candidato a substituir o atual, e sim o proprio motor atual como o
# usuario de fato ve).
PRODUCTION_RULES_ENGINE = "rules_production"
BOOTSTRAP_METRICS = ("precision_at_5", "precision_at_10", "recall_at_5", "recall_at_10", "ndcg_at_10", "mrr")


@dataclass(frozen=True)
class CaseEngineRecord:
    case_id: str
    engine: str
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float | None
    recall_at_10: float | None
    ndcg_at_10: float | None
    mrr: float
    invalid_id_rate: float | None
    execution_time_s: float
    reproducibility_exact_match_rate: float
    reproducibility_mean_top_k_jaccard: float
    predicted_top_10: tuple[str, ...]
    llm_reported_model_id: str | None = None
    """Identificador exato do modelo LLM que respondeu, conforme devolvido
    pelo servidor (ver ``engines.base.RankResult.reported_model_id``).
    ``None`` para motores que nao sao o LLM, ou quando o servidor nao
    informa o campo."""


@dataclass(frozen=True)
class EngineAggregate:
    engine: str
    n_cases: int
    mean_precision_at_5: float
    mean_precision_at_10: float
    mean_recall_at_5: float | None
    mean_recall_at_10: float | None
    mean_ndcg_at_10: float | None
    mean_mrr: float
    mean_invalid_id_rate: float | None
    mean_execution_time_s: float
    mean_reproducibility_exact_match_rate: float
    distinct_llm_reported_model_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationReport:
    metadata: RunMetadata
    provenance: ProvenanceReport
    records: tuple[CaseEngineRecord, ...]
    aggregates: tuple[EngineAggregate, ...]
    bootstrap_vs_baseline: dict[str, dict[str, PairedBootstrapResult]]


def _rank_with_engine(
    engine_name: str,
    context: ContractContext,
    risks: list[RiskItem],
    config: RunConfig,
    embedding_backend: EmbeddingBackend | None,
    llm_client: LLMClient | None,
) -> RankResult:
    if engine_name == "rules":
        return rules_engine.rank(context, risks)
    if engine_name == PRODUCTION_RULES_ENGINE:
        return rules_engine.rank_as_shown_in_production(context, risks)
    if engine_name == "bm25":
        return bm25_engine.rank(context, risks, config.bm25)
    if engine_name == "embedding":
        assert embedding_backend is not None
        return embedding_engine.rank(context, risks, embedding_backend)
    if engine_name == "llm":
        return llm_engine.rank(context, risks, config.llm, client=llm_client)
    raise ValueError(f"Motor desconhecido: {engine_name!r}")


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def evaluate_case_with_engine(
    case: EvalCase,
    engine_name: str,
    risks: list[RiskItem],
    known_ids: frozenset[str],
    config: RunConfig,
    embedding_backend: EmbeddingBackend | None = None,
    llm_client: LLMClient | None = None,
) -> CaseEngineRecord:
    def _rank_once() -> RankResult:
        return _rank_with_engine(engine_name, case.context, risks, config, embedding_backend, llm_client)

    started = time.perf_counter()
    result = _rank_once()
    elapsed = time.perf_counter() - started

    relevant = case.gabarito_set
    predicted = result.ranked_ids

    reproducibility = measure_reproducibility(
        lambda: _rank_once().ranked_ids,
        repeats=config.reproducibility_repeats,
    )

    return CaseEngineRecord(
        case_id=case.case_id,
        engine=engine_name,
        precision_at_5=precision_at_k(predicted, relevant, 5),
        precision_at_10=precision_at_k(predicted, relevant, 10),
        recall_at_5=recall_at_k(predicted, relevant, 5),
        recall_at_10=recall_at_k(predicted, relevant, 10),
        ndcg_at_10=ndcg_at_k(predicted, relevant, config.ndcg_k),
        mrr=mean_reciprocal_rank(predicted, relevant),
        invalid_id_rate=invalid_id_rate(result.raw_ids_before_validation, known_ids),
        execution_time_s=elapsed,
        reproducibility_exact_match_rate=reproducibility.exact_match_rate,
        reproducibility_mean_top_k_jaccard=reproducibility.mean_top_k_jaccard,
        predicted_top_10=tuple(predicted[:10]),
        llm_reported_model_id=result.reported_model_id,
    )


def aggregate_engine_records(engine_name: str, records: list[CaseEngineRecord]) -> EngineAggregate:
    distinct_model_ids = tuple(
        sorted({r.llm_reported_model_id for r in records if r.llm_reported_model_id is not None})
    )
    return EngineAggregate(
        engine=engine_name,
        n_cases=len(records),
        mean_precision_at_5=statistics.fmean(r.precision_at_5 for r in records),
        mean_precision_at_10=statistics.fmean(r.precision_at_10 for r in records),
        mean_recall_at_5=_mean_or_none([r.recall_at_5 for r in records]),
        mean_recall_at_10=_mean_or_none([r.recall_at_10 for r in records]),
        mean_ndcg_at_10=_mean_or_none([r.ndcg_at_10 for r in records]),
        mean_mrr=statistics.fmean(r.mrr for r in records),
        mean_invalid_id_rate=_mean_or_none([r.invalid_id_rate for r in records]),
        mean_execution_time_s=statistics.fmean(r.execution_time_s for r in records),
        mean_reproducibility_exact_match_rate=statistics.fmean(r.reproducibility_exact_match_rate for r in records),
        distinct_llm_reported_model_ids=distinct_model_ids,
    )


def _bootstrap_metric_order(config: RunConfig) -> tuple[str, ...]:
    """Métrica principal (``config.primary_metric``, pré-registrada antes dos
    dados reais) primeiro; as demais, exploratórias, na ordem de sempre."""
    others = tuple(metric for metric in BOOTSTRAP_METRICS if metric != config.primary_metric)
    return (config.primary_metric, *others)


def compute_bootstrap_vs_baseline(
    records: list[CaseEngineRecord],
    config: RunConfig,
    baseline_name: str = BASELINE_ENGINE,
) -> dict[str, dict[str, PairedBootstrapResult]]:
    by_engine_case = {(record.engine, record.case_id): record for record in records}
    # rules_production e informativo (o motor atual com os cortes de
    # producao), nao um candidato a substituir o motor atual -- por isso
    # nao entra na comparacao via bootstrap.
    engine_names = sorted({record.engine for record in records} - {baseline_name, PRODUCTION_RULES_ENGINE})

    comparisons: dict[str, dict[str, PairedBootstrapResult]] = {}
    for engine_name in engine_names:
        case_ids = sorted(
            case_id
            for (name, case_id) in by_engine_case
            if name == engine_name and (baseline_name, case_id) in by_engine_case
        )
        metric_results: dict[str, PairedBootstrapResult] = {}
        for metric in _bootstrap_metric_order(config):
            diffs = []
            for case_id in case_ids:
                baseline_value = getattr(by_engine_case[(baseline_name, case_id)], metric)
                candidate_value = getattr(by_engine_case[(engine_name, case_id)], metric)
                if baseline_value is None or candidate_value is None:
                    continue
                diffs.append(candidate_value - baseline_value)
            if len(diffs) >= 2:
                metric_results[metric] = paired_bootstrap_ci(
                    diffs,
                    confidence=config.bootstrap_confidence,
                    n_resamples=config.bootstrap_resamples,
                    seed=config.seed,
                )
        comparisons[engine_name] = metric_results
    return comparisons


def run_evaluation(
    risks_path: str | Path,
    config: RunConfig | None = None,
    include_real_cases: bool = True,
    include_synthetic_cases: bool = False,
    extra_cases: list[EvalCase] | None = None,
    embedding_backend: EmbeddingBackend | None = None,
    llm_client: LLMClient | None = None,
) -> EvaluationReport:
    config = config or RunConfig()
    risks = load_risks(risks_path)
    known_ids = frozenset(risk.id for risk in risks)

    cases: list[EvalCase] = list(extra_cases or [])
    if include_real_cases:
        cases += load_real_cases()
    if include_synthetic_cases:
        cases += load_synthetic_cases()

    provenance = split_by_provenance(cases, risks)
    engines = config.enabled_engines()

    if "embedding" in engines and embedding_backend is None:
        embedding_backend = build_default_backend(config.embedding)

    # rules_production (os mesmos cortes que o app mostra ao usuario) e
    # calculado automaticamente sempre que "rules" (ranking completo) roda,
    # para que o relatorio traga as duas visoes lado a lado -- ver
    # PRODUCTION_RULES_ENGINE.
    engines_to_run = list(engines)
    if "rules" in engines:
        engines_to_run.append(PRODUCTION_RULES_ENGINE)

    records: list[CaseEngineRecord] = []
    for case in provenance.main_eval_cases:
        for engine_name in engines_to_run:
            records.append(
                evaluate_case_with_engine(
                    case,
                    engine_name,
                    risks,
                    known_ids,
                    config,
                    embedding_backend=embedding_backend,
                    llm_client=llm_client,
                )
            )

    aggregates = tuple(
        aggregate_engine_records(engine_name, [r for r in records if r.engine == engine_name])
        for engine_name in engines_to_run
        if any(r.engine == engine_name for r in records)
    )
    bootstrap_vs_baseline = (
        compute_bootstrap_vs_baseline(records, config) if BASELINE_ENGINE in engines else {}
    )

    return EvaluationReport(
        metadata=collect_run_metadata(config),
        provenance=provenance,
        records=tuple(records),
        aggregates=aggregates,
        bootstrap_vs_baseline=bootstrap_vs_baseline,
    )


def _report_to_json_dict(report: EvaluationReport) -> dict:
    return {
        "metadata": report.metadata.to_dict(),
        "primary_metric": report.metadata.config.get("primary_metric"),
        "exploratory_metrics": [m for m in BOOTSTRAP_METRICS if m != report.metadata.config.get("primary_metric")],
        "provenance": {
            "main_eval_case_ids": [case.case_id for case in report.provenance.main_eval_cases],
            "excluded_leaked_case_ids": [case.case_id for case in report.provenance.excluded_leaked_cases],
            "cases_with_unknown_gabarito_ids": report.provenance.cases_with_unknown_gabarito_ids,
            "methodology_warnings": report.provenance.methodology_warnings,
        },
        "records": [asdict(record) for record in report.records],
        "aggregates": [asdict(aggregate) for aggregate in report.aggregates],
        "bootstrap_vs_baseline": {
            engine_name: {metric: asdict(result) for metric, result in metric_results.items()}
            for engine_name, metric_results in report.bootstrap_vs_baseline.items()
        },
    }


def write_report(report: EvaluationReport, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report.json").write_text(
        json.dumps(_report_to_json_dict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(report.records[0]).keys()) if report.records else [])
        if report.records:
            writer.writeheader()
            for record in report.records:
                row = asdict(record)
                row["predicted_top_10"] = ";".join(row["predicted_top_10"])
                writer.writerow(row)

    (output_dir / "summary.md").write_text(_render_markdown_summary(report), encoding="utf-8")
    return output_dir


def _render_markdown_summary(report: EvaluationReport) -> str:
    primary_metric = report.metadata.config.get("primary_metric", "recall_at_10")
    lines = ["# Resumo da avaliacao comparativa de motores", ""]
    lines.append(
        f"**Metrica principal (pre-registrada antes dos dados reais): `{primary_metric}` "
        "(Revocacao@10).** As demais metricas desta tabela sao exploratorias: ajudam a "
        "entender o comportamento dos motores, mas nao decidem a troca do motor atual."
    )
    lines.append("")
    lines.append(f"Casos na avaliacao principal: {len(report.provenance.main_eval_cases)}")
    lines.append(f"Casos excluidos (usados na construcao/revisao da base): {len(report.provenance.excluded_leaked_cases)}")
    if report.provenance.methodology_warnings:
        lines.append("")
        lines.append("### Avisos de metodologia de anotacao (nao bloqueiam a avaliacao)")
        for case_id, warnings in sorted(report.provenance.methodology_warnings.items()):
            for warning in warnings:
                lines.append(f"- `{case_id}`: {warning}")
    lines.append("")
    lines.append(
        "| Motor | Casos | P@5 (exploratoria) | P@10 (exploratoria) | R@5 (exploratoria) | "
        "**R@10 (principal)** | nDCG@10 (exploratoria) | MRR (exploratoria) | IDs invalidos | "
        "Tempo (s) | Reprodutibilidade |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for aggregate in report.aggregates:
        label = aggregate.engine
        if aggregate.engine == PRODUCTION_RULES_ENGINE:
            label += " (cortes de producao, informativo)"
        lines.append(
            "| {engine} | {n} | {p5:.3f} | {p10:.3f} | {r5} | **{r10}** | {ndcg} | {mrr:.3f} | {invalid} | {time:.4f} | {repro:.3f} |".format(
                engine=label,
                n=aggregate.n_cases,
                p5=aggregate.mean_precision_at_5,
                p10=aggregate.mean_precision_at_10,
                r5=f"{aggregate.mean_recall_at_5:.3f}" if aggregate.mean_recall_at_5 is not None else "-",
                r10=f"{aggregate.mean_recall_at_10:.3f}" if aggregate.mean_recall_at_10 is not None else "-",
                ndcg=f"{aggregate.mean_ndcg_at_10:.3f}" if aggregate.mean_ndcg_at_10 is not None else "-",
                mrr=aggregate.mean_mrr,
                invalid=f"{aggregate.mean_invalid_id_rate:.3f}" if aggregate.mean_invalid_id_rate is not None else "-",
                time=aggregate.mean_execution_time_s,
                repro=aggregate.mean_reproducibility_exact_match_rate,
            )
        )
        if aggregate.distinct_llm_reported_model_ids:
            lines.append(
                f"  - Identificador exato do modelo (reportado pelo servidor): "
                f"{', '.join(aggregate.distinct_llm_reported_model_ids)}"
            )
    lines.append("")
    lines.append(
        f"`{PRODUCTION_RULES_ENGINE}` e o motor atual com os mesmos cortes que o app mostra "
        "ao usuario (`minimum_score` e limite por categoria); nao e um motor candidato e "
        "fica fora do bootstrap abaixo -- esta na tabela so para contexto."
    )
    lines.append("")
    lines.append(f"## Bootstrap pareado vs. motor atual (regras) -- metrica principal: `{primary_metric}`")
    lines.append("")
    for engine_name, metric_results in report.bootstrap_vs_baseline.items():
        lines.append(f"### {engine_name} vs. {BASELINE_ENGINE}")
        for metric, result in metric_results.items():
            tag = "principal" if metric == primary_metric else "exploratoria"
            lines.append(
                f"- {metric} ({tag}): diferenca media = {result.observed_mean_diff:+.4f}, "
                f"IC {result.confidence:.0%} = [{result.ci_lower:+.4f}, {result.ci_upper:+.4f}], "
                f"p (bootstrap) ~= {result.p_value_two_sided:.4f}, n={result.n_cases}"
            )
        lines.append("")
    return "\n".join(lines)
