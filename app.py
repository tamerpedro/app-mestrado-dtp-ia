from __future__ import annotations

import base64
import re
from collections import Counter, defaultdict
from html import escape
import io
from pathlib import Path

import streamlit as st
from openpyxl import Workbook

from src.ai_provider import AIProviderError, FakeProvider, build_provider, load_ai_config
from src.docx_exporter import to_docx
from src.domain import (
    ACTION_STATUSES,
    CATEGORIES,
    CONTRACT_TYPES,
    CRITICALITIES,
    DEFAULT_MODALITY,
    MODALITIES,
    STRATEGIES,
    action_status_label,
    category_label,
    contract_type_label,
    criticality_label,
    impact_label,
    modality_label,
    probability_label,
    risk_level_label,
    strategy_label,
)
from src.eval_log import (
    DISCARD_REASONS,
    EvaluationLog,
    PairOutcome,
    SuggestionLogEntry,
    edit_ratio,
    outcome_for_pair,
    save_evaluation_log,
)
from src.eval_mode import is_modo_avaliacao, library_hash
from src.i18n import DEFAULT_LANGUAGE, LANGUAGES, normalize_language, t
from src.exporters import EXPORT_FIELDS, row_to_export_dict, selected_rows, to_csv, to_latex
from src.models import ActionItem, ContractContext, MatrixRow
from src.pairing import ORIGIN_AI, ORIGIN_LIBRARY, CategoryPair, build_category_pairs
from src.risk_generator import PROMPT_VERSION
from src.risk_library import load_risks, save_matrix_row_to_library
from src.scoring import IMPACT_OPTIONS, PROBABILITY_OPTIONS, canonical_scale, risk_level


DATA_PATH = Path("data/riscos_base.csv")
CANDIDATES_PATH = Path("data/candidatos_ia.csv")
LOG_DIR = Path("data/avaliacao")
LOGO_PATH = Path("assets/dataprev-logo.png")
LIKERT_SCALE = [1, 2, 3, 4, 5]

# Mesmo corte padrao de src.suggestions.suggest_risks: nao altera a logica de pontuacao,
# so reaproveita o mesmo limiar para escolher o risco de biblioteca de cada par.
DEFAULT_MINIMUM_SCORE = 2

_SLOT_ID_RE = re.compile(r"S\d{2}")


def resolve_language() -> str:
    """Idioma da sessao: parametro ?lang= na primeira visita, depois o seletor."""
    if "lang" not in st.session_state:
        st.session_state.lang = normalize_language(st.query_params.get("lang", DEFAULT_LANGUAGE))
    return normalize_language(st.session_state.lang)


LANG = resolve_language()
LANG_CHANGED = st.session_state.get("_rendered_lang") != LANG
st.session_state._rendered_lang = LANG


def tr(key: str, **params) -> str:
    return t(key, LANG, **params)


def label_in(label_func):
    """Adapta um rotulo de dominio ao idioma atual, para uso em format_func."""
    return lambda value: label_func(value, LANG)


def select(label: str, options: list, *, key: str, format_func, index: int = 0, **kwargs):
    """selectbox cujas opcoes acompanham o idioma sem perder o valor escolhido.

    O navegador nao redesenha o texto das opcoes quando so o ``format_func``
    muda. Por isso o widget ganha uma chave por idioma (``key@lang``) e o valor
    escolhido fica guardado em ``st.session_state[key]``, de onde o widget do
    outro idioma e reinicializado na troca.
    """
    widget_key = f"{key}@{LANG}"
    if LANG_CHANGED or widget_key not in st.session_state:
        stored = st.session_state.get(key)
        st.session_state[widget_key] = stored if stored in options else options[index]
    value = st.selectbox(label, options, format_func=format_func, key=widget_key, **kwargs)
    st.session_state[key] = value
    return value


def library_text(widget, label: str, default: str, *, key: str) -> str:
    """Campo de texto preenchido pela biblioteca de riscos.

    Enquanto o usuario nao editar, o texto acompanha o idioma (o ``default``
    muda com ele). Depois de editado, o texto do usuario e mantido.
    """
    seed_key = f"{key}__seed"
    seed = st.session_state.get(seed_key)
    untouched = key in st.session_state and st.session_state[key] == seed
    if key not in st.session_state or (untouched and default != seed):
        st.session_state[key] = default
    st.session_state[seed_key] = default
    return widget(label, key=key)


st.set_page_config(page_title=tr("app.title"), layout="wide")


def apply_dataprev_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --dtp-blue: #005ca9;
            --dtp-blue-dark: #003c71;
            --dtp-cyan: #00a3e0;
            --dtp-green: #79b829;
            --dtp-yellow: #f5c400;
            --dtp-border: rgba(0, 163, 224, .28);
            --dtp-soft: rgba(0, 92, 169, .12);
        }

        .block-container {
            padding-top: 2.35rem;
            max-width: 1280px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--dtp-border);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: 0;
        }

        .dtp-sidebar-brand {
            border-left: 5px solid var(--dtp-green);
            border-bottom: 1px solid var(--dtp-border);
            padding: .75rem 0 .9rem .9rem;
            margin: -.35rem 0 1.15rem 0;
        }

        .dtp-sidebar-brand strong {
            display: block;
            color: var(--dtp-cyan);
            font-size: 1.35rem;
            line-height: 1.1;
        }

        .dtp-sidebar-brand span {
            color: inherit;
            font-size: .82rem;
            opacity: .82;
        }

        .dtp-hero {
            position: relative;
            padding: 1.2rem 1.35rem 1.05rem 1.35rem;
            border: 1px solid var(--dtp-border);
            border-left: 7px solid var(--dtp-blue);
            border-radius: 8px;
            background:
                linear-gradient(90deg, rgba(0, 92, 169, .20), rgba(0, 163, 224, .07)),
                var(--dtp-soft);
            margin-bottom: 1.2rem;
        }

        .dtp-hero-main {
            display: flex;
            align-items: center;
            gap: 1.15rem;
            min-width: 0;
        }

        .dtp-logo-wrap {
            flex: 0 0 auto;
            width: clamp(72px, 9vw, 112px);
            aspect-ratio: 1.14;
            display: grid;
            place-items: center;
            padding: .45rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, .92);
            box-shadow: 0 10px 24px rgba(0, 60, 113, .12);
        }

        .dtp-logo-wrap img {
            width: 100%;
            height: auto;
            display: block;
        }

        .dtp-hero-copy {
            min-width: 0;
        }

        .dtp-kicker {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            color: var(--dtp-cyan);
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        .dtp-kicker::before {
            content: "";
            width: .7rem;
            height: .7rem;
            border-radius: 2px;
            background: linear-gradient(135deg, var(--dtp-green), var(--dtp-yellow));
        }

        .dtp-hero h1 {
            margin: .35rem 0 .25rem 0;
            font-size: clamp(2rem, 3.5vw, 3.15rem);
            line-height: 1.05;
            letter-spacing: 0;
        }

        .dtp-hero p {
            margin: 0;
            max-width: 860px;
            opacity: .86;
        }

        .dtp-status-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .7rem;
            margin: 1rem 0 0 0;
        }

        .dtp-status {
            border-top: 3px solid var(--dtp-cyan);
            background: rgba(255, 255, 255, .045);
            border-radius: 6px;
            padding: .7rem .75rem;
            min-height: 4.3rem;
        }

        .dtp-status span {
            display: block;
            font-size: .72rem;
            opacity: .72;
            margin-bottom: .25rem;
        }

        .dtp-status strong {
            display: block;
            font-size: 1rem;
            line-height: 1.25;
        }

        .dtp-panel-title {
            display: flex;
            align-items: center;
            gap: .55rem;
            margin: .25rem 0 .75rem 0;
        }

        .dtp-panel-title::before {
            content: "";
            width: .35rem;
            height: 1.45rem;
            border-radius: 999px;
            background: var(--dtp-green);
        }

        .dtp-section-label {
            margin: 1rem 0 .35rem 0;
            padding-top: .35rem;
            border-top: 1px solid var(--dtp-border);
            color: var(--dtp-cyan);
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .45rem;
            border-bottom: 1px solid var(--dtp-border);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: .7rem 1rem;
            letter-spacing: 0;
        }

        .stTabs [aria-selected="true"] {
            color: var(--dtp-cyan);
            border-bottom: 3px solid var(--dtp-green);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-color: var(--dtp-border);
            border-radius: 6px;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--dtp-cyan);
            color: var(--dtp-cyan);
        }

        [data-testid="stMetric"] {
            border-left: 4px solid var(--dtp-green);
            padding-left: .75rem;
        }

        @media (max-width: 900px) {
            .dtp-status-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 560px) {
            .dtp-status-grid {
                grid-template-columns: 1fr;
            }
            .dtp-hero {
                padding: 1rem;
            }
            .dtp-hero-main {
                align-items: flex-start;
                gap: .8rem;
            }
            .dtp-logo-wrap {
                width: 64px;
                padding: .35rem;
            }
            .dtp-kicker {
                font-size: .72rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_panel_title(title: str) -> None:
    st.markdown(f'<h3 class="dtp-panel-title">{escape(title)}</h3>', unsafe_allow_html=True)


def image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_app_header(context: ContractContext, total_pairs: int, sem_par_count: int) -> None:
    logo_uri = image_to_data_uri(LOGO_PATH)
    logo_html = (
        f'<div class="dtp-logo-wrap"><img src="{logo_uri}" alt="{escape(tr("app.logo_alt"))}"></div>'
        if logo_uri
        else ""
    )
    st.markdown(
        f"""
        <section class="dtp-hero">
            <div class="dtp-hero-main">
                {logo_html}
                <div class="dtp-hero-copy">
                    <div class="dtp-kicker">{escape(tr("app.kicker"))}</div>
                    <h1>{escape(tr("app.title"))}</h1>
                    <p>{escape(tr("app.subtitle"))}</p>
                </div>
            </div>
            <div class="dtp-status-grid">
                <div class="dtp-status"><span>{escape(tr("header.type"))}</span><strong>{escape(contract_type_label(context.tipo_contratacao, LANG))}</strong></div>
                <div class="dtp-status"><span>{escape(tr("header.criticality"))}</span><strong>{escape(criticality_label(context.criticidade, LANG))}</strong></div>
                <div class="dtp-status"><span>{escape(tr("header.pairs"))}</span><strong>{escape(tr("header.pairs_value", total=total_pairs, sem_par=sem_par_count))}</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    st.markdown(f'<div class="dtp-section-label">{escape(label)}</div>', unsafe_allow_html=True)


apply_dataprev_theme()


def context_state_key(context: ContractContext) -> str:
    return "|".join(
        [
            context.objeto,
            context.tipo_contratacao,
            str(context.valor_estimado),
            context.criticidade,
            context.prazo,
            context.modalidade,
            context.contexto,
        ]
    )


def _clear_blind_widget_state() -> None:
    """Limpa os widgets de S01..S99 para que uma nova rodada de pares (apos
    mudar o contexto) nao herde textos editados de uma rodada anterior."""
    for state_key in list(st.session_state.keys()):
        if _SLOT_ID_RE.search(state_key):
            del st.session_state[state_key]


def ensure_pairing_result(context: ContractContext, risks: list, provider, lang: str):
    """Monta (e cacheia por contexto) os pares biblioteca+IA por categoria.

    So chama a IA de novo quando o contexto muda (nao o idioma: o texto da
    biblioteca se adapta ao idioma na propria renderizacao, via
    ``risks_by_id``; o texto gerado pela IA fica congelado no idioma em que
    foi gerado). Evita gerar (e sortear a ordem) a cada interacao do
    Streamlit, e evita perder edicoes do usuario so por trocar o idioma.
    """
    key = context_state_key(context)
    if st.session_state.get("pairing_context_key") != key:
        _clear_blind_widget_state()
        failures: list[dict] = []
        result = build_category_pairs(
            risks,
            context,
            provider,
            minimum_score=DEFAULT_MINIMUM_SCORE,
            lang=lang,
            on_ai_failure=lambda categoria, erro: failures.append({"categoria": categoria, "erro": erro}),
        )
        st.session_state.pairing_context_key = key
        st.session_state.pairing_result = result
        st.session_state.pairing_failures = failures
    return st.session_state.pairing_result, st.session_state.pairing_failures


def row_text_blob(row: MatrixRow) -> str:
    return "||".join([row.risco, row.causa, row.consequencia, row.acao_preventiva, row.acao_contingencia])


def render_language_selector() -> None:
    st.radio(
        tr("sidebar.language"),
        list(LANGUAGES),
        format_func=LANGUAGES.get,
        horizontal=True,
        key="lang",
    )
    st.query_params["lang"] = st.session_state.lang


def build_context() -> ContractContext:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="dtp-sidebar-brand">
                <strong>Dataprev</strong>
                <span>{escape(tr("sidebar.brand"))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_language_selector()
        st.text_input(tr("process.code"), help=tr("process.help"), key="codigo_processo")
        st.header(tr("sidebar.contract"))
        objeto = st.text_area(tr("field.object"), value=tr("field.object_default"), key="ctx_objeto")
        tipo = select(
            tr("field.type"),
            CONTRACT_TYPES,
            format_func=label_in(contract_type_label),
            key="ctx_tipo",
        )
        valor = st.number_input(
            tr("field.value"),
            min_value=0.0,
            step=1000.0,
            help=tr("help.value"),
            key="ctx_valor",
        )
        criticidade = select(
            tr("field.criticality"),
            CRITICALITIES,
            index=1,
            format_func=label_in(criticality_label),
            help=tr("help.criticality"),
            key="ctx_criticidade",
        )
        prazo_meses = st.number_input(
            tr("field.term"),
            min_value=1,
            value=12,
            step=1,
            help=tr("help.term"),
            key="ctx_prazo",
        )
        modalidade = select(
            tr("field.modality"),
            MODALITIES,
            index=MODALITIES.index(DEFAULT_MODALITY),
            format_func=label_in(modality_label),
            help=tr("help.modality"),
            key="ctx_modalidade",
        )
        contexto = st.text_area(
            tr("field.context"),
            value=tr("field.context_default"),
            help=tr("help.context"),
            key="ctx_contexto",
        )
    return ContractContext(
        objeto=objeto,
        tipo_contratacao=tipo,
        valor_estimado=valor,
        criticidade=criticidade,
        # Mantido em PT: entra na chave de estado e no texto analisado pelas sugestoes.
        prazo=f"{prazo_meses} meses",
        modalidade=modalidade,
        contexto=contexto,
    )


FINAL_MATRIX_COLUMNS = {
    "id": "col.id",
    "risco": "col.risk",
    "categoria": "col.category",
    "causa": "col.cause",
    "consequencia": "col.consequence",
    "probabilidade": "col.probability",
    "impacto": "col.impact",
    "nivel": "col.level",
    "estrategia": "col.strategy",
    "acao_preventiva": "col.preventive",
    "acao_contingencia": "col.contingency",
    "justificativa": "col.rationale",
}


def final_matrix_table(rows: list[MatrixRow]) -> list[dict[str, str]]:
    table = []
    for row in rows:
        data = row_to_export_dict(row, LANG)
        table.append({tr(FINAL_MATRIX_COLUMNS[field]): data[field] for field in EXPORT_FIELDS})
    return table


def rows_to_xlsx(rows: list[MatrixRow]) -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = t("export.sheet_title", DEFAULT_LANGUAGE)
    worksheet.append(EXPORT_FIELDS)
    for row in selected_rows(rows):
        data = row_to_export_dict(row)
        worksheet.append([data[field] for field in EXPORT_FIELDS])
    stream = io.BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def ensure_manual_rows() -> None:
    if "manual_rows" not in st.session_state:
        st.session_state.manual_rows = []


def next_manual_id() -> str:
    ensure_manual_rows()
    return f"MAN{len(st.session_state.manual_rows) + 1:03d}"


def safe_index(options: list[str], value: str, default: int = 0) -> int:
    return options.index(value) if value in options else default


def add_manual_risk_form() -> None:
    ensure_manual_rows()
    with st.expander(tr("manual.expander"), expanded=False):
        with st.form("manual_risk_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                manual_id = st.text_input(tr("field.id"), value=next_manual_id())
                categoria = select(
                    tr("field.category"), CATEGORIES, format_func=label_in(category_label), key="manual_categoria"
                )
            with col2:
                probabilidade = select(
                    tr("field.probability"),
                    PROBABILITY_OPTIONS,
                    index=2,
                    format_func=label_in(probability_label),
                    key="manual_probabilidade",
                )
                impacto = select(
                    tr("field.impact"), IMPACT_OPTIONS, index=2, format_func=label_in(impact_label), key="manual_impacto"
                )
            with col3:
                estrategia = select(
                    tr("field.strategy"), STRATEGIES, format_func=label_in(strategy_label), key="manual_estrategia"
                )

            risco = st.text_input(tr("field.risk"))
            causa = st.text_area(tr("field.cause"))
            consequencia = st.text_area(tr("manual.consequence"))
            preventiva = st.text_area(tr("manual.preventive"))
            contingencia = st.text_area(tr("manual.contingency"))
            justificativa = st.text_area(tr("manual.notes"))
            submitted = st.form_submit_button(tr("manual.submit"))

        if submitted and risco.strip():
            st.session_state.manual_rows.append(
                MatrixRow(
                    id=manual_id.strip() or next_manual_id(),
                    risco=risco.strip(),
                    categoria=categoria,
                    causa=causa.strip(),
                    consequencias=[consequencia.strip()] if consequencia.strip() else [],
                    probabilidade=probabilidade,
                    impacto=impacto,
                    nivel=risk_level(probabilidade, impacto),
                    estrategia=estrategia,
                    acoes_preventivas=[ActionItem(preventiva.strip())] if preventiva.strip() else [],
                    acoes_contingencia=[ActionItem(contingencia.strip())] if contingencia.strip() else [],
                    justificativa=justificativa.strip() or tr("manual.default_note"),
                    tags=["manual"],
                )
            )
            st.success(tr("manual.added", id=manual_id))
        elif submitted:
            st.warning(tr("manual.missing_title"))


def edit_text_items(risk_key: str, slot: str, base_items: list[str]) -> list[str]:
    """``slot`` e fixo (compoe as chaves de estado); o rotulo vem de ``item.<slot>``."""
    label = tr(f"item.{slot}")
    count_key = f"{risk_key}_{slot}_count"
    deleted_key = f"{risk_key}_{slot}_deleted"
    if count_key not in st.session_state:
        st.session_state[count_key] = max(1, len(base_items))
    if deleted_key not in st.session_state:
        st.session_state[deleted_key] = []
    if st.button(tr("item.add", item=label.lower()), key=f"{risk_key}_{slot}_add"):
        st.session_state[count_key] += 1

    items: list[str] = []
    deleted = set(st.session_state[deleted_key])
    for index in range(st.session_state[count_key]):
        if index in deleted:
            continue
        default = base_items[index] if index < len(base_items) else ""
        col1, col2 = st.columns([5, 1])
        with col1:
            value = library_text(st.text_area, f"{label} {index + 1}", default, key=f"{risk_key}_{slot}_{index}")
        with col2:
            st.write("")
            st.write("")
            if index > 0 and st.button(tr("item.delete"), key=f"{risk_key}_{slot}_delete_{index}"):
                st.session_state[deleted_key].append(index)
                st.rerun()
        if value.strip():
            items.append(value.strip())
    return items


def edit_action_items(risk_key: str, slot: str, base_actions: list[ActionItem]) -> list[ActionItem]:
    label = tr(f"item.{slot}")
    count_key = f"{risk_key}_{slot}_count"
    deleted_key = f"{risk_key}_{slot}_deleted"
    if count_key not in st.session_state:
        st.session_state[count_key] = max(1, len(base_actions))
    if deleted_key not in st.session_state:
        st.session_state[deleted_key] = []
    if st.button(tr("item.add", item=label.lower()), key=f"{risk_key}_{slot}_add"):
        st.session_state[count_key] += 1

    actions: list[ActionItem] = []
    deleted = set(st.session_state[deleted_key])
    for index in range(st.session_state[count_key]):
        if index in deleted:
            continue
        base = base_actions[index] if index < len(base_actions) else ActionItem("")
        col1, col2, col3, col4 = st.columns([3, 1, 2, 1])
        with col1:
            descricao = library_text(
                st.text_area, f"{label} {index + 1}", base.descricao, key=f"{risk_key}_{slot}_desc_{index}"
            )
        with col2:
            situacao = select(
                tr("field.status"),
                ACTION_STATUSES,
                index=safe_index(ACTION_STATUSES, base.situacao),
                format_func=label_in(action_status_label),
                key=f"{risk_key}_{slot}_sit_{index}",
            )
        with col3:
            responsavel = st.text_input(
                tr("field.owner"),
                value=base.responsavel,
                key=f"{risk_key}_{slot}_resp_{index}",
            )
        with col4:
            st.write("")
            st.write("")
            if index > 0 and st.button(tr("item.delete"), key=f"{risk_key}_{slot}_delete_{index}"):
                st.session_state[deleted_key].append(index)
                st.rerun()
        if descricao.strip():
            actions.append(ActionItem(descricao.strip(), situacao=situacao, responsavel=responsavel.strip()))
    return actions


def edit_rows(rows: list[MatrixRow], context: ContractContext) -> list[MatrixRow]:
    """Edicao dos riscos manuais (inclusao continua igual; nada aqui e as cegas)."""
    edited_by_index: dict[int, MatrixRow] = {}
    for category, indexed_rows in grouped_row_indexes(rows):
        render_section_label(f"{category_label(category, LANG)} ({len(indexed_rows)})")
        with st.container():
            for index, row in indexed_rows:
                risk_key = f"{row.id}_{index}"
                with st.expander(f"{row.id} - {row.risco}", expanded=row.selecionado):
                    selecionado = st.checkbox(tr("review.include"), value=row.selecionado, key=f"sel_{risk_key}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        probabilidade = select(
                            tr("field.probability"),
                            PROBABILITY_OPTIONS,
                            index=PROBABILITY_OPTIONS.index(canonical_scale(row.probabilidade)),
                            format_func=label_in(probability_label),
                            key=f"prob_{risk_key}",
                        )
                    with col2:
                        impacto = select(
                            tr("field.impact"),
                            IMPACT_OPTIONS,
                            index=IMPACT_OPTIONS.index(canonical_scale(row.impacto)),
                            format_func=label_in(impact_label),
                            key=f"impacto_{risk_key}",
                        )
                    with col3:
                        nivel = risk_level(probabilidade, impacto)
                        st.metric(tr("field.level"), risk_level_label(nivel, LANG))

                    categoria = select(
                        tr("review.category"),
                        CATEGORIES,
                        index=safe_index(CATEGORIES, row.categoria),
                        format_func=label_in(category_label),
                        key=f"cat_{risk_key}",
                    )
                    estrategia = select(
                        tr("field.strategy"),
                        STRATEGIES,
                        index=safe_index(STRATEGIES, row.estrategia),
                        format_func=label_in(strategy_label),
                        key=f"estrategia_{risk_key}",
                    )

                    risco = library_text(st.text_input, tr("field.risk"), row.risco, key=f"risco_{risk_key}")
                    causa = library_text(st.text_area, tr("field.cause"), row.causa, key=f"causa_{risk_key}")
                    render_section_label(tr("section.consequences"))
                    consequencias = edit_text_items(risk_key, "consequence", row.consequencias)
                    render_section_label(tr("section.preventive"))
                    preventivas = edit_action_items(risk_key, "preventive", row.acoes_preventivas)
                    render_section_label(tr("section.contingency"))
                    contingencias = edit_action_items(risk_key, "contingency", row.acoes_contingencia)
                    justificativa = library_text(
                        st.text_area, tr("review.rationale"), row.justificativa, key=f"just_{risk_key}"
                    )
                    edited_row = MatrixRow(
                        id=row.id,
                        risco=risco,
                        categoria=categoria,
                        causa=causa,
                        consequencias=consequencias,
                        probabilidade=probabilidade,
                        impacto=impacto,
                        nivel=nivel,
                        estrategia=estrategia,
                        acoes_preventivas=preventivas,
                        acoes_contingencia=contingencias,
                        justificativa=justificativa,
                        selecionado=selecionado,
                        tags=row.tags,
                    )
                    if "manual" in row.tags:
                        disabled = is_modo_avaliacao()
                        if st.button(tr("review.save_library"), key=f"save_library_{risk_key}", disabled=disabled):
                            result = save_matrix_row_to_library(DATA_PATH, edited_row, context, lang=LANG)
                            if result.saved:
                                st.success(tr("library.saved", id=result.risk_id))
                                st.rerun()
                            elif result.risk_id:
                                st.info(tr("library.exists", id=result.risk_id))
                            else:
                                st.error(tr("library.error", detail=result.detail or result.message))
                        if disabled:
                            st.caption(tr("library.save_disabled"))
                    edited_by_index[index] = edited_row
    return [edited_by_index[index] for index in range(len(rows))]


def grouped_row_indexes(rows: list[MatrixRow]) -> list[tuple[str, list[tuple[int, MatrixRow]]]]:
    grouped: dict[str, list[tuple[int, MatrixRow]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row.categoria or "planejamento"].append((index, row))

    def sort_key(item: tuple[str, list[tuple[int, MatrixRow]]]) -> tuple[int, str]:
        category = item[0]
        if category in CATEGORIES:
            return (CATEGORIES.index(category), category)
        return (len(CATEGORIES), category)

    return sorted(grouped.items(), key=sort_key)


def render_blind_pair(
    pair: CategoryPair, par_index: int, decisions: dict[str, dict], risks_by_id: dict[str, object]
) -> None:
    """Renderiza um par (ou uma sugestao isolada, se ``sem_par``) e preenche
    ``decisions[slot_id]`` com a decisao do usuario para essa sugestao."""
    sem_par_suffix = tr("pair.sem_par_suffix")
    suffix = f" — {sem_par_suffix}" if pair.sem_par else ""
    render_section_label(f"{category_label(pair.categoria, LANG)}{suffix}")

    for posicao, suggestion in enumerate(pair.suggestions, start=1):
        slot_key = suggestion.slot_id
        row = suggestion.row
        # Sugestao de biblioteca: o texto acompanha o idioma atual (consulta o
        # RiskItem original); sugestao de IA fica congelada no idioma gerado.
        source_risk = risks_by_id.get(row.id) if suggestion.origin == ORIGIN_LIBRARY else None
        if source_risk is not None:
            default_risco = source_risk.text("titulo", LANG)
            default_causa = source_risk.text("causa", LANG)
            default_consequencias = [source_risk.text("consequencia", LANG)]
            default_preventivas = [ActionItem(source_risk.text("acao_preventiva", LANG))]
            default_contingencias = [ActionItem(source_risk.text("acao_contingencia", LANG))]
        else:
            default_risco = row.risco
            default_causa = row.causa
            default_consequencias = row.consequencias
            default_preventivas = row.acoes_preventivas
            default_contingencias = row.acoes_contingencia

        with st.expander(f"{slot_key} - {default_risco}", expanded=False):
            incluido = st.checkbox(tr("review.include"), value=False, key=f"incl_{slot_key}")
            col1, col2, col3 = st.columns(3)
            with col1:
                probabilidade = select(
                    tr("field.probability"),
                    PROBABILITY_OPTIONS,
                    index=PROBABILITY_OPTIONS.index(canonical_scale(row.probabilidade)),
                    format_func=label_in(probability_label),
                    key=f"prob_{slot_key}",
                )
            with col2:
                impacto = select(
                    tr("field.impact"),
                    IMPACT_OPTIONS,
                    index=IMPACT_OPTIONS.index(canonical_scale(row.impacto)),
                    format_func=label_in(impact_label),
                    key=f"impacto_{slot_key}",
                )
            with col3:
                nivel = risk_level(probabilidade, impacto)
                st.metric(tr("field.level"), risk_level_label(nivel, LANG))

            risco = library_text(st.text_input, tr("field.risk"), default_risco, key=f"risco_{slot_key}")
            causa = library_text(st.text_area, tr("field.cause"), default_causa, key=f"causa_{slot_key}")
            render_section_label(tr("section.consequences"))
            consequencias = edit_text_items(slot_key, "consequence", default_consequencias)
            render_section_label(tr("section.preventive"))
            preventivas = edit_action_items(slot_key, "preventive", default_preventivas)
            render_section_label(tr("section.contingency"))
            contingencias = edit_action_items(slot_key, "contingency", default_contingencias)

            motivo_descarte = None
            if not incluido:
                motivo_codigo = select(
                    tr("discard.reason"),
                    ["", *DISCARD_REASONS],
                    format_func=lambda code: tr(f"discard.reason.{code}") if code else tr("discard.reason.none"),
                    key=f"motivo_{slot_key}",
                )
                motivo_descarte = motivo_codigo or None

            edited_row = MatrixRow(
                id=slot_key,
                risco=risco,
                categoria=pair.categoria,
                causa=causa,
                consequencias=consequencias,
                probabilidade=probabilidade,
                impacto=impacto,
                nivel=nivel,
                estrategia=row.estrategia,
                acoes_preventivas=preventivas,
                acoes_contingencia=contingencias,
                selecionado=incluido,
                tags=row.tags,
            )
            # Linha de base no idioma atual (nao a de geracao): troca de idioma sozinha
            # nao deve contar como edicao do usuario.
            baseline_row = MatrixRow(
                id=row.id,
                risco=default_risco,
                categoria=pair.categoria,
                causa=default_causa,
                consequencias=default_consequencias,
                probabilidade=row.probabilidade,
                impacto=row.impacto,
                nivel=row.nivel,
                estrategia=row.estrategia,
                acoes_preventivas=default_preventivas,
                acoes_contingencia=default_contingencias,
                tags=row.tags,
            )
            ratio = edit_ratio(row_text_blob(baseline_row), row_text_blob(edited_row))
            decisions[slot_key] = {
                "row": edited_row,
                "origin": suggestion.origin,
                "categoria": pair.categoria,
                "par": par_index,
                "posicao": posicao,
                "incluido": incluido,
                "editado": ratio > 0.0,
                "proporcao_edicao": ratio,
                "motivo_descarte": motivo_descarte,
            }


context = build_context()
try:
    risks = load_risks(DATA_PATH)
except ValueError as exc:
    st.error(tr("error.load_library", detail=exc))
    st.stop()
risks_by_id = {risk.id: risk for risk in risks}

ai_config = load_ai_config()
ai_config_error: str | None = None
try:
    provider = build_provider(ai_config)
except AIProviderError as exc:
    provider = FakeProvider(fail=True)
    ai_config_error = str(exc)

pairing_result, pairing_failures = ensure_pairing_result(context, risks, provider, LANG)
sem_par_count = sum(1 for pair in pairing_result.pairs if pair.sem_par)

render_app_header(context, len(pairing_result.pairs), sem_par_count)

if ai_config_error:
    st.warning(tr("ai.unavailable", detail=ai_config_error))
elif pairing_failures:
    first_detail = pairing_failures[0]["erro"]
    st.warning(tr("ai.partial_failure", count=len(pairing_failures), detail=first_detail))

tab1, tab2, tab3 = st.tabs([tr("tab.suggestions"), tr("tab.review"), tr("tab.export")])

pairing_decisions: dict[str, dict] = {}

with tab1:
    render_panel_title(tr("panel.suggested"))
    total_suggestions = sum(len(pair.suggestions) for pair in pairing_result.pairs)
    col1, col2, col3 = st.columns(3)
    col1.metric(tr("metric.pairs_total"), total_suggestions)
    col2.metric(tr("metric.categories"), len(pairing_result.pairs))
    col3.metric(tr("metric.sem_par"), sem_par_count)

    if not pairing_result.pairs:
        st.info(tr("empty.suggested"))
    else:
        for par_index, pair in enumerate(pairing_result.pairs, start=1):
            render_blind_pair(pair, par_index, pairing_decisions, risks_by_id)

with tab2:
    render_panel_title(tr("panel.review"))
    add_manual_risk_form()
    edited_manual_rows = edit_rows(st.session_state.manual_rows, context)

with tab3:
    render_panel_title(tr("panel.final"))

    slot_origin = {
        suggestion.slot_id: suggestion.origin for pair in pairing_result.pairs for suggestion in pair.suggestions
    }
    all_rows_for_export = [decision["row"] for decision in pairing_decisions.values()] + edited_manual_rows
    selected = selected_rows(all_rows_for_export)

    render_section_label(tr("reveal.title"))
    reveal_table = []
    for row in selected:
        origin = slot_origin.get(row.id, "manual")
        origin_key = origin if origin in (ORIGIN_LIBRARY, ORIGIN_AI) else "manual"
        reveal_table.append(
            {
                tr("col.id"): row.id,
                tr("reveal.origin_column"): tr(f"reveal.{origin_key}"),
                tr("col.risk"): row.risco,
                tr("col.category"): category_label(row.categoria, LANG),
            }
        )
    if reveal_table:
        st.dataframe(reveal_table, width="stretch", hide_index=True)
    else:
        st.info(tr("empty.list"))

    library_count = sum(1 for row in selected if slot_origin.get(row.id) == ORIGIN_LIBRARY)
    ai_count = sum(1 for row in selected if slot_origin.get(row.id) == ORIGIN_AI)
    manual_count = sum(1 for row in selected if row.id not in slot_origin)
    edited_count = sum(
        1
        for row in selected
        if row.id in pairing_decisions and pairing_decisions[row.id]["editado"]
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(tr("reveal.library"), library_count)
    col2.metric(tr("reveal.ai"), ai_count)
    col3.metric(tr("reveal.manual_metric"), manual_count)
    col4.metric(tr("reveal.edited"), edited_count)

    outcome_counts = Counter(
        outcome_for_pair(
            pair.sem_par,
            any(pairing_decisions.get(s.slot_id, {}).get("incluido") and s.origin == ORIGIN_LIBRARY for s in pair.suggestions),
            any(pairing_decisions.get(s.slot_id, {}).get("incluido") and s.origin == ORIGIN_AI for s in pair.suggestions),
        )
        for pair in pairing_result.pairs
    )
    outcome_table = [
        {tr("outcome.column_label"): tr(f"outcome.{code}"), tr("outcome.count_label"): outcome_counts.get(code, 0)}
        for code in ["so_biblioteca", "so_ia", "ambos", "nenhum", "sem_par"]
        if outcome_counts.get(code, 0) > 0
    ]
    if outcome_table:
        st.dataframe(outcome_table, width="stretch", hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric(tr("metric.selected"), len(selected))
    col2.metric(tr("metric.preventive"), sum(len(row.acoes_preventivas) for row in selected))
    col3.metric(tr("metric.contingency"), sum(len(row.acoes_contingencia) for row in selected))
    st.dataframe(final_matrix_table(selected), width="stretch", hide_index=True)
    if LANG != DEFAULT_LANGUAGE:
        st.caption(tr("export.pt_only"))

    render_section_label(tr("likert.header"))
    utilidade = st.radio(tr("likert.usefulness"), LIKERT_SCALE, horizontal=True, key="likert_utilidade")
    confianca = st.radio(tr("likert.confidence"), LIKERT_SCALE, horizontal=True, key="likert_confianca")
    esforco = st.radio(tr("likert.effort"), LIKERT_SCALE, horizontal=True, key="likert_esforco")
    comentario = st.text_area(tr("likert.comment"), key="likert_comentario")

    codigo_processo = (st.session_state.get("codigo_processo") or "").strip()
    if not codigo_processo:
        st.warning(tr("likert.missing_process_code"))

    if st.button(tr("likert.submit"), disabled=not codigo_processo):
        log = EvaluationLog(
            codigo_processo=codigo_processo,
            idioma=LANG,
            provedor_ia=ai_config.provider,
            modelo_ia=ai_config.model,
            versao_prompt=PROMPT_VERSION,
            hash_biblioteca=library_hash(DATA_PATH),
            semente=pairing_result.seed,
        )
        # Se a IA nao estava configurada, as falhas por categoria sao todas a
        # mesma causa raiz: so a entrada global entra no log.
        log.falhas_ia = [{"categoria": "*", "erro": ai_config_error}] if ai_config_error else list(pairing_failures)

        library_target = CANDIDATES_PATH if is_modo_avaliacao() else DATA_PATH
        for par_index, pair in enumerate(pairing_result.pairs, start=1):
            library_chosen = False
            ai_chosen = False
            for suggestion in pair.suggestions:
                decision = pairing_decisions.get(suggestion.slot_id)
                if decision is None:
                    continue
                log.sugestoes.append(
                    SuggestionLogEntry(
                        par=decision["par"],
                        categoria=decision["categoria"],
                        origem=decision["origin"],
                        posicao=decision["posicao"],
                        slot_id=suggestion.slot_id,
                        escolhido=decision["incluido"],
                        editado=decision["editado"],
                        proporcao_edicao=decision["proporcao_edicao"],
                        motivo_descarte=decision["motivo_descarte"],
                    )
                )
                if decision["incluido"]:
                    if suggestion.origin == ORIGIN_LIBRARY:
                        library_chosen = True
                    else:
                        ai_chosen = True
                        save_matrix_row_to_library(library_target, decision["row"], context, lang=LANG)
            log.pares.append(
                PairOutcome(
                    par=par_index,
                    categoria=pair.categoria,
                    desfecho=outcome_for_pair(pair.sem_par, library_chosen, ai_chosen),
                )
            )

        log.quantidade_riscos_manuais = len(st.session_state.manual_rows)
        log.respostas_finais = {
            "utilidade": utilidade,
            "confianca": confianca,
            "esforco": esforco,
            "comentario": comentario.strip(),
        }
        save_evaluation_log(log, directory=LOG_DIR)
        st.success(tr("likert.saved"))

    csv_content = to_csv(selected)
    latex_content = to_latex(selected)
    docx_content = to_docx(selected, context)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button(tr("download.csv"), csv_content, "matriz_riscos.csv", "text/csv")
    with col2:
        st.download_button(
            tr("download.excel"),
            rows_to_xlsx(selected),
            "matriz_riscos.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col3:
        st.download_button(tr("download.latex"), latex_content, "matriz_riscos.tex", "text/plain")
    with col4:
        st.download_button(
            tr("download.word"),
            docx_content,
            "mapa_de_gerenciamento_de_riscos.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
