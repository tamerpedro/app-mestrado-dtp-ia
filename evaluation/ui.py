"""Interface Streamlit para gerenciar casos, anotar gabaritos e rodar a
avaliacao comparativa dos motores de sugestao.

App **separado** de ``app.py``/``src/`` -- nao altera nada ali, nem importa
nada de ``src`` alem do que ja e publico e usado em modo leitura (``ContractContext``,
rotulos/listas de ``src.domain``, ``load_risks``). Reutiliza o schema
(``cases/schema.py``), o loader (``cases/loader.py``) e o runner
(``runner.py``) existentes -- esta interface nao reimplementa validacao de
caso nem logica de execucao dos motores, so monta a entrada e exibe a
saida.

Uso local (NUNCA publique esta interface com dados reais em uma instancia
acessivel pela rede -- ver o aviso em ``evaluation/README.md``):

    streamlit run evaluation/ui.py --server.address 127.0.0.1

As funcoes que nao comecam com ``render_`` nem sao ``main`` nao chamam a API
do Streamlit e podem ser testadas diretamente (ver ``evaluation/tests/test_ui.py``).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

# ``streamlit run evaluation/ui.py`` executa este arquivo como script, nao
# como parte do pacote ``evaluation`` -- o Python so coloca o diretorio do
# script em sys.path, nao a raiz do repositorio. Garantimos a raiz aqui para
# que os imports absolutos de ``src`` e ``evaluation`` abaixo funcionem
# independentemente de como/de onde o Streamlit e invocado.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.domain import (
    CONTRACT_TYPES,
    CRITICALITIES,
    MODALITIES,
    category_label,
    contract_type_label,
    criticality_label,
    modality_label,
)
from src.models import RiskItem
from src.risk_library import load_risks

from evaluation.__main__ import DEFAULT_RISKS_PATH
from evaluation.cases.loader import DEFAULT_REAL_CASES_DIR, load_cases_from_dir
from evaluation.cases.schema import CaseFormatError, EvalCase, case_from_dict
from evaluation.config import (
    EXPLORATORY_METRICS,
    PRIMARY_METRIC,
    EmbeddingConfig,
    LLMConfig,
    LLMConfigError,
    RunConfig,
)
from evaluation.runner import BASELINE_ENGINE, EvaluationReport, run_evaluation

PREDATES_RISK_MAP_OPTIONS: dict[bool | None, str] = {
    None: "Nao informado",
    True: "Sim",
    False: "Nao",
}


# --------------------------------------------------------------------------
# Logica pura (sem chamadas a st.*): testavel sem um runtime Streamlit.
# --------------------------------------------------------------------------


def new_draft_case() -> dict:
    """Um caso em edicao em memoria. O gabarito comeca vazio -- o schema
    exige ``gabarito_ids`` nao vazio, entao um caso novo so pode ser
    persistido depois de marcar ao menos 1 risco na aba Gabarito."""
    return {
        "case_id": "",
        "objeto": "",
        "tipo_contratacao": CONTRACT_TYPES[0],
        "valor_estimado": 0.0,
        "criticidade": CRITICALITIES[0],
        "prazo": "",
        "modalidade": MODALITIES[0],
        "contexto": "",
        "used_to_build_base": False,
        "context_predates_risk_map": None,
        "annotator_count": None,
        "provenance_notes": "",
        "gabarito_ids": [],
    }


def case_to_draft(case: EvalCase) -> dict:
    return {
        "case_id": case.case_id,
        "objeto": case.context.objeto,
        "tipo_contratacao": case.context.tipo_contratacao,
        "valor_estimado": case.context.valor_estimado,
        "criticidade": case.context.criticidade,
        "prazo": case.context.prazo,
        "modalidade": case.context.modalidade,
        "contexto": case.context.contexto,
        "used_to_build_base": case.used_to_build_base,
        "context_predates_risk_map": case.context_predates_risk_map,
        "annotator_count": case.annotator_count,
        "provenance_notes": case.provenance_notes,
        "gabarito_ids": list(case.gabarito_ids),
    }


def draft_to_payload(draft: dict) -> dict:
    """Monta o JSON no formato que ``cases.schema.case_from_dict`` espera
    (ver ``evaluation/README.md``). Campos opcionais (``context_predates_risk_map``,
    ``annotator_count``) so entram quando informados -- o schema aceita a
    ausencia deles."""
    payload: dict = {
        "case_id": draft["case_id"],
        "context": {
            "objeto": draft["objeto"],
            "tipo_contratacao": draft["tipo_contratacao"],
            "valor_estimado": draft["valor_estimado"],
            "criticidade": draft["criticidade"],
            "prazo": draft["prazo"],
            "modalidade": draft["modalidade"],
            "contexto": draft["contexto"],
        },
        "gabarito_ids": list(draft["gabarito_ids"]),
        "used_to_build_base": bool(draft["used_to_build_base"]),
        "provenance_notes": draft.get("provenance_notes", ""),
    }
    if draft.get("context_predates_risk_map") is not None:
        payload["context_predates_risk_map"] = draft["context_predates_risk_map"]
    if draft.get("annotator_count"):
        payload["annotator_count"] = draft["annotator_count"]
    return payload


def case_file_path(case_id: str, cases_dir: Path = DEFAULT_REAL_CASES_DIR) -> Path:
    safe_id = "".join(char for char in case_id.strip() if char.isalnum() or char in "-_") or "caso"
    return cases_dir / f"{safe_id}.json"


def save_case(draft: dict, cases_dir: Path = DEFAULT_REAL_CASES_DIR) -> tuple[bool, str]:
    """Valida com ``cases.schema.case_from_dict`` (a mesma validacao usada
    pelo loader/runner -- nao duplicada aqui) e, se valida, persiste em
    ``<cases_dir>/<case_id>.json``. Devolve ``(ok, mensagem)``; nunca escreve
    um arquivo invalido."""
    if not draft["case_id"].strip():
        return False, "Informe um case_id antes de salvar."

    payload = draft_to_payload(draft)
    try:
        case_from_dict(payload, source_path="<evaluation/ui.py>")
    except CaseFormatError as exc:
        return False, str(exc)

    cases_dir.mkdir(parents=True, exist_ok=True)
    path = case_file_path(draft["case_id"], cases_dir)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True, f"Caso salvo em {path}"


def load_existing_cases(cases_dir: Path = DEFAULT_REAL_CASES_DIR) -> tuple[list[EvalCase], str | None]:
    """Mesmo loader usado pelo runner (``cases.loader.load_cases_from_dir``).
    Se algum arquivo estiver malformado, devolve a lista vazia e a mensagem
    de erro -- nao reimplementa uma leitura "tolerante" por fora do loader."""
    try:
        return load_cases_from_dir(cases_dir), None
    except CaseFormatError as exc:
        return [], str(exc)


def filter_risks(risks: Sequence[RiskItem], query: str) -> list[RiskItem]:
    normalized = query.strip().lower()
    if not normalized:
        return list(risks)
    return [
        risk
        for risk in risks
        if normalized in risk.id.lower()
        or normalized in risk.titulo.lower()
        or normalized in category_label(risk.categoria).lower()
    ]


def compute_found_missing(predicted_top_10: Sequence[str], gabarito: frozenset[str]) -> tuple[list[str], list[str]]:
    predicted_set = set(predicted_top_10)
    found = sorted(predicted_set & gabarito)
    missing = sorted(gabarito - predicted_set)
    return found, missing


def build_run_config(
    *,
    engines: tuple[str, ...],
    embedding_backend: str,
    llm_enabled: bool,
    llm_endpoint: str,
    llm_model: str,
    llm_allow_external: bool,
) -> RunConfig:
    """Pode levantar ``LLMConfigError`` (endpoint LLM nao local sem
    ``allow_external`` explicito) -- a mesma validacao de seguranca de
    ``evaluation/config.py``, nao duplicada aqui."""
    return RunConfig(
        engines=engines,
        embedding=EmbeddingConfig(backend=embedding_backend),
        llm=LLMConfig(
            enabled=llm_enabled,
            endpoint=llm_endpoint,
            model=llm_model,
            allow_external=llm_allow_external,
        ),
    )


# --------------------------------------------------------------------------
# Renderizacao (Streamlit) -- nao testada por import direto.
# --------------------------------------------------------------------------


def _init_state() -> None:
    if "draft_case" not in st.session_state:
        st.session_state.draft_case = new_draft_case()
    if "report" not in st.session_state:
        st.session_state.report = None
    if "report_embedding_backend" not in st.session_state:
        st.session_state.report_embedding_backend = None


def render_casos_tab() -> None:
    st.header("Casos")
    st.caption(
        "Cadastre a contratacao (campos de ContractContext) e os metadados de "
        "proveniencia. O gabarito (quais riscos aparecem no mapa real) e "
        "marcado na aba Gabarito."
    )

    draft = st.session_state.draft_case
    existing_cases, load_error = load_existing_cases()
    if load_error:
        st.error(
            f"Erro ao carregar casos existentes em {DEFAULT_REAL_CASES_DIR}: {load_error}. "
            "Corrija ou remova o arquivo indicado."
        )

    col_new, col_load = st.columns(2)
    with col_new:
        if st.button("Novo caso"):
            st.session_state.draft_case = new_draft_case()
            st.rerun()
    with col_load:
        if existing_cases:
            selected_id = st.selectbox(
                "Carregar caso existente para editar",
                options=[case.case_id for case in existing_cases],
                index=None,
                placeholder="Selecione um case_id...",
            )
            if selected_id and st.button("Carregar"):
                case = next(c for c in existing_cases if c.case_id == selected_id)
                st.session_state.draft_case = case_to_draft(case)
                st.rerun()

    if existing_cases:
        st.subheader("Casos existentes")
        st.dataframe(
            [
                {
                    "case_id": case.case_id,
                    "objeto": case.context.objeto[:80],
                    "tipo_contratacao": case.context.tipo_contratacao,
                    "criticidade": case.context.criticidade,
                    "riscos no gabarito": len(case.gabarito_ids),
                    "used_to_build_base": case.used_to_build_base,
                    "context_predates_risk_map": case.context_predates_risk_map,
                    "annotator_count": case.annotator_count,
                }
                for case in existing_cases
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Editar caso")
    is_saved_case = bool(draft["case_id"]) and any(c.case_id == draft["case_id"] for c in existing_cases)
    draft["case_id"] = st.text_input(
        "case_id",
        value=draft["case_id"],
        disabled=is_saved_case,
        help="Identificador unico (nome do arquivo em evaluation/cases/real/). "
        "Nao pode ser alterado depois de salvo.",
    )

    draft["objeto"] = st.text_area("Objeto da contratacao", value=draft["objeto"])
    draft["tipo_contratacao"] = st.selectbox(
        "Tipo de contratacao",
        options=CONTRACT_TYPES,
        index=CONTRACT_TYPES.index(draft["tipo_contratacao"]) if draft["tipo_contratacao"] in CONTRACT_TYPES else 0,
        format_func=contract_type_label,
    )
    draft["valor_estimado"] = st.number_input(
        "Valor estimado (R$)", min_value=0.0, value=float(draft["valor_estimado"]), step=1000.0
    )
    draft["criticidade"] = st.selectbox(
        "Criticidade",
        options=CRITICALITIES,
        index=CRITICALITIES.index(draft["criticidade"]) if draft["criticidade"] in CRITICALITIES else 0,
        format_func=criticality_label,
    )
    draft["prazo"] = st.text_input("Prazo", value=draft["prazo"])
    draft["modalidade"] = st.selectbox(
        "Modalidade",
        options=MODALITIES,
        index=MODALITIES.index(draft["modalidade"]) if draft["modalidade"] in MODALITIES else 0,
        format_func=modality_label,
    )
    draft["contexto"] = st.text_area("Contexto adicional", value=draft["contexto"])

    st.subheader("Proveniencia")
    draft["used_to_build_base"] = st.checkbox(
        "Este caso alimentou, revisou ou ajustou a biblioteca de riscos (texto, "
        "palavras-chave, pesos) ou os limiares de um motor (used_to_build_base)",
        value=draft["used_to_build_base"],
        help="Em caso de duvida, marque: casos assim sao excluidos da avaliacao principal.",
    )
    predates_keys = list(PREDATES_RISK_MAP_OPTIONS.keys())
    draft["context_predates_risk_map"] = st.selectbox(
        "O contexto vem de documentos anteriores ao mapa de riscos final?",
        options=predates_keys,
        index=predates_keys.index(draft["context_predates_risk_map"]),
        format_func=lambda key: PREDATES_RISK_MAP_OPTIONS[key],
    )
    draft["annotator_count"] = (
        st.number_input(
            "Numero de anotadores independentes do gabarito (0 = nao informado)",
            min_value=0,
            step=1,
            value=draft["annotator_count"] or 0,
        )
        or None
    )
    draft["provenance_notes"] = st.text_area("Notas de proveniencia (opcional)", value=draft["provenance_notes"])

    st.info(f"Riscos no gabarito: {len(draft['gabarito_ids'])} (edite na aba Gabarito).")

    if st.button("Salvar caso", type="primary"):
        ok, message = save_case(draft)
        if ok:
            st.success(message)
        else:
            st.warning(message)


def render_gabarito_tab(risks: list[RiskItem]) -> None:
    st.header("Gabarito")
    draft = st.session_state.draft_case

    if not draft["case_id"].strip():
        st.info("Crie ou carregue um caso na aba Casos antes de marcar o gabarito.")
        return

    st.warning(
        "Marque somente os riscos presentes no mapa de riscos REAL desta contratacao. "
        "Esta aba nunca mostra pontuacoes ou sugestoes de motor -- a anotacao deve ser "
        "feita sem acesso a saida de nenhum motor (idealmente por 2 anotadores "
        "independentes). Ver 'Proveniencia do contexto e do gabarito' em evaluation/README.md."
    )
    st.caption(f"Caso: `{draft['case_id']}`")

    query = st.text_input("Buscar por ID, titulo ou categoria", value="")
    filtered = filter_risks(risks, query)

    selected = set(draft["gabarito_ids"])
    for risk in filtered:
        checked = st.checkbox(
            f"{risk.id} -- {risk.titulo} ({category_label(risk.categoria)})",
            value=risk.id in selected,
            key=f"gabarito_{draft['case_id']}_{risk.id}",
        )
        if checked:
            selected.add(risk.id)
        else:
            selected.discard(risk.id)
    if not filtered:
        st.caption("Nenhum risco encontrado para esta busca.")

    draft["gabarito_ids"] = sorted(selected)
    st.write(f"**{len(selected)}** risco(s) marcado(s).")

    if st.button("Salvar gabarito", type="primary"):
        ok, message = save_case(draft)
        if ok:
            st.success(message)
        else:
            st.warning(message)


def _render_summary_table(report: EvaluationReport) -> None:
    st.subheader("Resumo por motor")
    st.caption(
        f"Metrica principal (pre-registrada antes dos dados reais): **{PRIMARY_METRIC}** "
        f"(Revocacao@10). Exploratorias: {', '.join(EXPLORATORY_METRICS)}."
    )
    rows = [
        {
            "Motor": aggregate.engine,
            "Casos": aggregate.n_cases,
            "P@5": round(aggregate.mean_precision_at_5, 3),
            "P@10": round(aggregate.mean_precision_at_10, 3),
            "R@5": round(aggregate.mean_recall_at_5, 3) if aggregate.mean_recall_at_5 is not None else None,
            "R@10 (principal)": (
                round(aggregate.mean_recall_at_10, 3) if aggregate.mean_recall_at_10 is not None else None
            ),
            "nDCG@10": round(aggregate.mean_ndcg_at_10, 3) if aggregate.mean_ndcg_at_10 is not None else None,
            "MRR": round(aggregate.mean_mrr, 3),
            "IDs invalidos": (
                round(aggregate.mean_invalid_id_rate, 3) if aggregate.mean_invalid_id_rate is not None else None
            ),
            "Tempo (s)": round(aggregate.mean_execution_time_s, 4),
            "Reprodutibilidade": round(aggregate.mean_reproducibility_exact_match_rate, 3),
        }
        for aggregate in report.aggregates
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    for aggregate in report.aggregates:
        if aggregate.distinct_llm_reported_model_ids:
            st.caption(
                f"Identificador exato do modelo LLM reportado pelo servidor: "
                f"{', '.join(aggregate.distinct_llm_reported_model_ids)}"
            )


def _render_bootstrap_section(report: EvaluationReport) -> None:
    st.subheader("Bootstrap pareado vs. motor atual (regras)")
    if not report.bootstrap_vs_baseline:
        st.caption("Sem comparacao disponivel (motor 'rules' nao incluido, ou poucos casos para bootstrap).")
        return

    for engine_name, metric_results in report.bootstrap_vs_baseline.items():
        st.markdown(f"**{engine_name}** vs. **{BASELINE_ENGINE}**")
        if not metric_results:
            st.caption("Poucos casos pareados para bootstrap (minimo 2) -- adicione mais casos reais.")
            continue
        rows = [
            {
                "Metrica": metric,
                "Tipo": "principal" if metric == PRIMARY_METRIC else "exploratoria",
                "Diferenca media": round(result.observed_mean_diff, 4),
                f"IC {result.confidence:.0%}": f"[{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]",
                "p (bootstrap)": round(result.p_value_two_sided, 4),
                "n casos": result.n_cases,
            }
            for metric, result in metric_results.items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_methodology_warnings(report: EvaluationReport) -> None:
    if not report.provenance.methodology_warnings:
        return
    st.subheader("Avisos de metodologia de anotacao")
    st.caption("Nao bloqueiam a avaliacao -- ver 'Proveniencia do contexto e do gabarito' em evaluation/README.md.")
    for case_id, warnings in sorted(report.provenance.methodology_warnings.items()):
        for warning in warnings:
            st.warning(f"`{case_id}`: {warning}")


def _render_found_missing(report: EvaluationReport) -> None:
    st.subheader("Riscos encontrados e perdidos por caso e motor (top 10)")
    gabarito_by_case = {case.case_id: case.gabarito_set for case in report.provenance.main_eval_cases}
    for record in report.records:
        gabarito = gabarito_by_case.get(record.case_id, frozenset())
        found, missing = compute_found_missing(record.predicted_top_10, gabarito)
        with st.expander(f"{record.case_id} -- {record.engine}"):
            st.write(f"Encontrados: {', '.join(found) if found else '-'}")
            st.write(f"Perdidos: {', '.join(missing) if missing else '-'}")


def render_avaliacao_tab() -> None:
    st.header("Avaliacao")

    st.subheader("Motores")
    st.caption("`rules_production` (cortes de producao) e incluido automaticamente junto de `rules`.")
    include_rules = st.checkbox("rules (+ rules_production)", value=True)
    include_bm25 = st.checkbox("bm25", value=True)
    include_embedding = st.checkbox("embedding", value=True)

    embedding_backend = "sentence-transformers"
    if include_embedding:
        embedding_backend = st.radio(
            "Backend de embeddings",
            options=["sentence-transformers", "hashing"],
            index=0,
            help="'hashing' e um fallback local deterministico sem modelo real -- use so para testar a interface.",
        )
        if embedding_backend == "hashing":
            st.warning(
                "Backend 'hashing' selecionado: NAO e um embedding semantico real. Os "
                "numeros do motor `embedding` nesta execucao NAO sao finais nem "
                "representativos -- nunca use para decidir a troca do motor atual."
            )

    include_llm = st.checkbox("llm (desligado por padrao)", value=False)
    llm_endpoint = "http://localhost:11434/v1/chat/completions"
    llm_model = "llama3.1:8b-instruct"
    llm_allow_external = False
    if include_llm:
        st.warning(
            "O motor LLM envia o contexto da contratacao e os riscos candidatos ao "
            "endpoint configurado. Mantenha um endpoint local; so marque 'permitir "
            "endpoint externo' se entender e aceitar enviar dados a um servico fora "
            "desta maquina."
        )
        llm_endpoint = st.text_input("Endpoint LLM (compativel com API de chat da OpenAI)", value=llm_endpoint)
        llm_model = st.text_input("Modelo (tag pedida ao servidor)", value=llm_model)
        llm_allow_external = st.checkbox("Permitir endpoint NAO local (allow_external)", value=False)

    include_synthetic = st.checkbox(
        "Incluir caso sintetico de exemplo (so demonstracao -- nunca use para decidir a troca do motor)",
        value=False,
    )

    if st.button("Rodar avaliacao", type="primary"):
        engines: list[str] = []
        if include_rules:
            engines.append("rules")
        if include_bm25:
            engines.append("bm25")
        if include_embedding:
            engines.append("embedding")
        if include_llm:
            engines.append("llm")

        if not engines:
            st.error("Selecione ao menos um motor.")
        else:
            try:
                config = build_run_config(
                    engines=tuple(engines),
                    embedding_backend=embedding_backend,
                    llm_enabled=include_llm,
                    llm_endpoint=llm_endpoint,
                    llm_model=llm_model,
                    llm_allow_external=llm_allow_external,
                )
            except LLMConfigError as exc:
                st.error(str(exc))
            else:
                with st.spinner("Rodando avaliacao..."):
                    try:
                        report = run_evaluation(
                            risks_path=DEFAULT_RISKS_PATH,
                            config=config,
                            include_real_cases=True,
                            include_synthetic_cases=include_synthetic,
                        )
                    except Exception as exc:  # motores externos (ex.: LLM) podem falhar em runtime
                        st.error(f"Falha ao rodar a avaliacao: {exc}")
                        report = None
                if report is not None:
                    st.session_state.report = report
                    st.session_state.report_embedding_backend = embedding_backend if include_embedding else None

    report: EvaluationReport | None = st.session_state.get("report")
    if report is None:
        st.info("Rode a avaliacao para ver os resultados.")
        return

    if st.session_state.get("report_embedding_backend") == "hashing":
        st.warning(
            "Este relatorio usou o backend 'hashing' para embeddings -- os numeros do "
            "motor `embedding` NAO sao finais nem representativos."
        )

    if not report.records:
        st.warning(
            "Nenhum caso na avaliacao principal. Cadastre casos reais com gabarito "
            "preenchido (abas Casos e Gabarito), ou marque 'Incluir caso sintetico' acima."
        )
        return

    _render_summary_table(report)
    _render_bootstrap_section(report)
    _render_methodology_warnings(report)
    _render_found_missing(report)


def main() -> None:
    st.set_page_config(page_title="Avaliacao de motores de sugestao", layout="wide")
    _init_state()

    st.title("Avaliacao comparativa dos motores de sugestao de riscos")
    st.caption(
        "Interface separada do app principal (nao altera app.py nem src/). "
        "NUNCA use dados reais de contratacoes com esta interface publicada em "
        "uma instancia acessivel pela rede -- ver evaluation/README.md."
    )

    risks = load_risks(DEFAULT_RISKS_PATH)

    tab_casos, tab_gabarito, tab_avaliacao = st.tabs(["Casos", "Gabarito", "Avaliacao"])
    with tab_casos:
        render_casos_tab()
    with tab_gabarito:
        render_gabarito_tab(risks)
    with tab_avaliacao:
        render_avaliacao_tab()


if __name__ == "__main__":
    main()
