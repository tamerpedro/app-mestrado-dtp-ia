from __future__ import annotations

import base64
import re
from collections import defaultdict
from html import escape
import io
from pathlib import Path

import streamlit as st
from openpyxl import Workbook

from src.ai_provider import AIProviderError, FakeProvider, build_provider, load_ai_config
from src.case_store import get_or_generate_case_ai_risks, list_case_files, load_case
from src.catalog_mix import (
    ORIGIN_CATALOG,
    CategoryItems,
    build_category_mix,
    generate_ai_risks_for_all_categories,
)
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
from src.eval_log import edit_ratio
from src.eval_mode import is_modo_avaliacao
from src.i18n import DEFAULT_LANGUAGE, LANGUAGES, normalize_language, t
from src.exporters import EXPORT_FIELDS, row_to_export_dict, selected_rows, to_csv, to_latex
from src.models import ActionItem, ContractContext, MatrixRow
from src.risk_library import load_risks, save_matrix_row_to_library
from src.scoring import IMPACT_OPTIONS, PROBABILITY_OPTIONS, canonical_scale, risk_level


DATA_PATH = Path("data/riscos_base.csv")
LOGO_PATH = Path("assets/dataprev-logo.png")
NO_CASE = "__nenhum__"

_SLOT_ID_RE = re.compile(r"S\d{2,}")


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
    muda com ele). Depois de editado, o texto do usuario e mantido - mesmo
    se o item sumir e voltar a aparecer entre rodadas (ex.: desmarcar e
    marcar de novo na Revisao). O Streamlit limpa o estado de um widget que
    nao e desenhado numa rodada; por isso guardamos uma copia "sticky" fora
    do ciclo de vida do widget, para restaurar quando ele reaparecer.
    """
    sticky = st.session_state.setdefault("_sticky_text", {})
    if key not in st.session_state and key in sticky:
        st.session_state[key] = sticky[key]
    seed_key = f"{key}__seed"
    seed = st.session_state.get(seed_key)
    untouched = key in st.session_state and st.session_state[key] == seed
    if key not in st.session_state or (untouched and default != seed):
        st.session_state[key] = default
    st.session_state[seed_key] = default
    value = widget(label, key=key)
    sticky[key] = value
    return value


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


def render_app_header(context: ContractContext, total_items: int, total_categories: int) -> None:
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
                <div class="dtp-status"><span>{escape(tr("header.items"))}</span><strong>{escape(tr("header.items_value", total=total_items, categories=total_categories))}</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    st.markdown(f'<div class="dtp-section-label">{escape(label)}</div>', unsafe_allow_html=True)


apply_dataprev_theme()


def _clear_blind_widget_state() -> None:
    """Limpa os widgets de S01..S99 (e a copia "sticky" deles, usada por
    ``library_text`` para sobreviver a desmarcar/marcar de novo) para que uma
    nova rodada de itens (apos mudar o contexto) nao herde textos editados de
    uma rodada anterior."""
    for state_key in list(st.session_state.keys()):
        if _SLOT_ID_RE.search(state_key):
            del st.session_state[state_key]
    sticky = st.session_state.get("_sticky_text")
    if sticky:
        for sticky_key in list(sticky.keys()):
            if _SLOT_ID_RE.search(sticky_key):
                del sticky[sticky_key]


def maybe_generate_mix(
    context: ContractContext,
    risks: list,
    provider,
    lang: str,
    generate_clicked: bool,
    selected_case_path: Path | None,
):
    """So monta a lista cega catalogo+IA (e chama a IA, ou reusa o cache do
    caso) quando o usuario clica em "Gerar sugestoes" - nunca so por abrir a
    tela ou editar um campo.

    O resultado fica em sessao ate o proximo clique (preserva selecoes e
    edicoes do usuario entre reruns do Streamlit, inclusive ao trocar o
    idioma: o texto do catalogo se adapta via ``risks_by_id`` na propria
    renderizacao; o texto gerado pela IA fica congelado no idioma em que
    foi gerado).
    """
    if generate_clicked:
        _clear_blind_widget_state()
        failures: list[dict] = []
        on_failure = lambda categoria, erro: failures.append({"categoria": categoria, "erro": erro})  # noqa: E731
        if selected_case_path is not None:
            ai_risks = get_or_generate_case_ai_risks(
                selected_case_path, lang, risks, context, provider, on_ai_failure=on_failure
            )
        else:
            ai_risks = generate_ai_risks_for_all_categories(
                risks, context, provider, lang=lang, on_ai_failure=on_failure
            )
        result = build_category_mix(risks, context, ai_risks, lang=lang)
        st.session_state.mix_result = result
        st.session_state.mix_failures = failures
    return st.session_state.get("mix_result"), st.session_state.get("mix_failures", [])


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


def _apply_case_to_context_fields(context: ContractContext) -> None:
    """Preenche os campos da contratacao com os de um caso salvo.

    So roda quando o caso selecionado muda (nao a cada rerun), para o
    usuario poder editar os campos depois sem que voltem ao valor do caso.
    Os campos com ``select()`` tem sua chave de widget limpa para forcar a
    resincronizacao com o novo valor nesta rerun.
    """
    st.session_state.ctx_objeto = context.objeto
    st.session_state.ctx_valor = context.valor_estimado
    st.session_state.ctx_contexto = context.contexto
    for key, value in [
        ("ctx_tipo", context.tipo_contratacao),
        ("ctx_criticidade", context.criticidade),
        ("ctx_modalidade", context.modalidade),
    ]:
        st.session_state[key] = value
        st.session_state.pop(f"{key}@{LANG}", None)
    try:
        st.session_state.ctx_prazo = int(str(context.prazo).split()[0])
    except (ValueError, IndexError):
        pass


def render_case_selector() -> Path | None:
    """Seletor "Caso" na barra lateral; aplica o contexto do caso escolhido
    uma unica vez (na troca), para o usuario poder editar depois."""
    case_files = list_case_files()
    options = [NO_CASE, *(str(path) for path in case_files)]
    labels = {NO_CASE: tr("case.none")}
    for path in case_files:
        nome, _ = load_case(path)
        labels[str(path)] = nome

    selected = select(
        tr("case.label"),
        options,
        format_func=lambda value: labels.get(value, value),
        key="selected_case",
    )

    if selected != NO_CASE and st.session_state.get("_applied_case") != selected:
        _, case_context = load_case(Path(selected))
        _apply_case_to_context_fields(case_context)
        st.session_state._applied_case = selected
    elif selected == NO_CASE:
        st.session_state._applied_case = None

    return Path(selected) if selected != NO_CASE else None


def build_context() -> tuple[ContractContext, bool, Path | None]:
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
        selected_case_path = render_case_selector()
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
        generate_clicked = st.button(tr("action.generate"), type="primary", key="btn_generate_suggestions")
    context = ContractContext(
        objeto=objeto,
        tipo_contratacao=tipo,
        valor_estimado=valor,
        criticidade=criticidade,
        # Mantido em PT: entra na chave de estado e no texto analisado pelas sugestoes.
        prazo=f"{prazo_meses} meses",
        modalidade=modalidade,
        contexto=contexto,
    )
    return context, generate_clicked, selected_case_path


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
            manual_id = next_manual_id()
            st.session_state.manual_rows.append(
                MatrixRow(
                    id=manual_id,
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
            responsavel = library_text(
                st.text_input,
                tr("field.owner"),
                base.responsavel,
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
    """Edicao dos riscos escolhidos na Revisao (itens as cegas + manuais).

    A chave de cada widget e ``row.id`` (ja unico: slot_id para itens as
    cegas, MANxxx para manuais) - nao a posicao na lista, que muda sempre
    que um item e marcado/desmarcado na aba Sugestoes. Usar a posicao faria
    o Streamlit "esquecer" as edicoes dos outros itens a cada mudanca.
    """
    edited_by_index: dict[int, MatrixRow] = {}
    for category, indexed_rows in grouped_row_indexes(rows):
        render_section_label(f"{category_label(category, LANG)} ({len(indexed_rows)})")
        with st.container():
            for index, row in indexed_rows:
                risk_key = row.id
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


def _blind_item_defaults(item, risks_by_id: dict[str, object]) -> dict:
    """Texto do item no idioma atual: para itens de catalogo, consulta o
    RiskItem original (acompanha o idioma); para IA, fica congelado no
    idioma em que foi gerado."""
    row = item.row
    source_risk = risks_by_id.get(row.id) if item.origin == ORIGIN_CATALOG else None
    if source_risk is not None:
        return {
            "risco": source_risk.text("titulo", LANG),
            "causa": source_risk.text("causa", LANG),
            "consequencias": [source_risk.text("consequencia", LANG)],
            "preventivas": [ActionItem(source_risk.text("acao_preventiva", LANG))],
            "contingencias": [ActionItem(source_risk.text("acao_contingencia", LANG))],
        }
    return {
        "risco": row.risco,
        "causa": row.causa,
        "consequencias": row.consequencias,
        "preventivas": row.acoes_preventivas,
        "contingencias": row.acoes_contingencia,
    }


def render_blind_category(
    category: CategoryItems, items_by_slot: dict[str, dict], risks_by_id: dict[str, object], ai_failed: bool
) -> None:
    """Renderiza a lista cega (catalogo+IA misturados) de uma categoria, so
    para escolher - nada aqui e editavel (a edicao fica na aba Revisao).

    O ID neutro, o titulo e a caixa "incluir" ficam visiveis sem precisar
    expandir; o expander mostra causa, consequencia, P×I e acoes so para
    leitura. Preenche ``items_by_slot[slot_id]`` com a decisao e os dados
    (linha-base, idioma atual) que a aba Revisao vai editar.
    """
    render_section_label(category_label(category.categoria, LANG))
    if ai_failed:
        st.caption(tr("category.ai_partial_warning"))

    for item in category.items:
        slot_key = item.slot_id
        row = item.row
        defaults = _blind_item_defaults(item, risks_by_id)

        col_check, col_title = st.columns([1, 11])
        with col_check:
            incluido = st.checkbox(
                tr("review.include"), value=False, key=f"incl_{slot_key}", label_visibility="collapsed"
            )
        with col_title:
            st.markdown(f"**{slot_key}** - {escape(defaults['risco'])}")

        with st.expander(tr("item.details"), expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(tr("field.probability"), probability_label(canonical_scale(row.probabilidade), LANG))
            with col2:
                st.metric(tr("field.impact"), impact_label(canonical_scale(row.impacto), LANG))
            with col3:
                nivel = risk_level(row.probabilidade, row.impacto)
                st.metric(tr("field.level"), risk_level_label(nivel, LANG))

            st.markdown(f"**{escape(tr('field.cause'))}:** {escape(defaults['causa'])}")
            consequencia_items = [text for text in defaults["consequencias"] if text]
            if consequencia_items:
                st.markdown(f"**{escape(tr('section.consequences'))}:**")
                for text in consequencia_items:
                    st.markdown(f"- {escape(text)}")
            preventiva_items = [action.descricao for action in defaults["preventivas"] if action.descricao]
            if preventiva_items:
                st.markdown(f"**{escape(tr('section.preventive'))}:**")
                for text in preventiva_items:
                    st.markdown(f"- {escape(text)}")
            contingencia_items = [action.descricao for action in defaults["contingencias"] if action.descricao]
            if contingencia_items:
                st.markdown(f"**{escape(tr('section.contingency'))}:**")
                for text in contingencia_items:
                    st.markdown(f"- {escape(text)}")

        items_by_slot[slot_key] = {
            "row": row,
            "origin": item.origin,
            "categoria": category.categoria,
            "defaults": defaults,
            "incluido": incluido,
        }


context, generate_clicked, selected_case_path = build_context()
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

mix_result, mix_failures = maybe_generate_mix(context, risks, provider, LANG, generate_clicked, selected_case_path)
categories = mix_result.categories if mix_result is not None else []
categories_with_ai_failure = {failure["categoria"] for failure in mix_failures}

render_app_header(context, sum(len(c.items) for c in categories), len(categories))

if ai_config_error:
    st.warning(tr("ai.unavailable", detail=ai_config_error))

tab1, tab2, tab3 = st.tabs([tr("tab.suggestions"), tr("tab.review"), tr("tab.export")])

items_by_slot: dict[str, dict] = {}

with tab1:
    render_panel_title(tr("panel.suggested"))
    if mix_result is None:
        st.info(tr("empty.not_generated"))
    else:
        col1, col2 = st.columns(2)
        col1.metric(tr("metric.items_total"), sum(len(c.items) for c in categories))
        col2.metric(tr("metric.categories"), len(categories))

        if not categories:
            st.info(tr("empty.suggested"))
        else:
            for category in categories:
                render_blind_category(
                    category, items_by_slot, risks_by_id, category.categoria in categories_with_ai_failure
                )

with tab2:
    render_panel_title(tr("panel.review"))
    ensure_manual_rows()

    chosen_slots = [slot for slot, info in items_by_slot.items() if info["incluido"]]
    manual_rows = st.session_state.manual_rows

    baseline_rows: list[MatrixRow] = []
    edited_rows: list[MatrixRow] = []

    if not chosen_slots and not manual_rows:
        st.info(tr("empty.review"))
    else:
        for slot in chosen_slots:
            info = items_by_slot[slot]
            defaults = info["defaults"]
            source_row = info["row"]
            baseline_rows.append(
                MatrixRow(
                    id=slot,
                    risco=defaults["risco"],
                    categoria=info["categoria"],
                    causa=defaults["causa"],
                    consequencias=defaults["consequencias"],
                    probabilidade=source_row.probabilidade,
                    impacto=source_row.impacto,
                    nivel=source_row.nivel,
                    estrategia=source_row.estrategia,
                    acoes_preventivas=defaults["preventivas"],
                    acoes_contingencia=defaults["contingencias"],
                    tags=source_row.tags,
                )
            )
        edited_rows = edit_rows(baseline_rows + manual_rows, context)

        # Medida de edicao: so para os itens escolhidos (nao para manuais), comparando
        # com a linha-base no idioma atual - trocar de idioma sozinho nao conta como edicao.
        review_decisions: dict[str, dict] = {}
        for baseline, edited in zip(baseline_rows, edited_rows[: len(baseline_rows)]):
            ratio = edit_ratio(row_text_blob(baseline), row_text_blob(edited))
            review_decisions[baseline.id] = {"editado": ratio > 0.0, "proporcao_edicao": ratio}
        st.session_state.review_decisions = review_decisions

    add_manual_risk_form()

with tab3:
    render_panel_title(tr("panel.final"))

    selected = selected_rows(edited_rows)

    st.caption(tr("export.final_pending"))

    col1, col2, col3 = st.columns(3)
    col1.metric(tr("metric.selected"), len(selected))
    col2.metric(tr("metric.preventive"), sum(len(row.acoes_preventivas) for row in selected))
    col3.metric(tr("metric.contingency"), sum(len(row.acoes_contingencia) for row in selected))
    st.dataframe(final_matrix_table(selected), width="stretch", hide_index=True)
    if LANG != DEFAULT_LANGUAGE:
        st.caption(tr("export.pt_only"))

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
