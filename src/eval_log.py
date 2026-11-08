"""Schema (v2) e gravacao do log de avaliacao, um JSON por processo.

Mede quanto do processo com catalogo e IA e adotado, qual a participacao da
IA na matriz final e se isso varia com a experiencia de quem elabora a
matriz - nao e uma comparacao direta entre a taxa da IA e a do catalogo.

Grava em ``data/avaliacao/`` (fora do controle de versao): metadados do
processo (caso, experiencia, idioma, provedor/modelo de IA, versao do
prompt, hash da biblioteca, semente), o detalhe de cada item mostrado na
aba Sugestoes (categoria, origem, se foi escolhido/editado, o texto como
foi mostrado e o texto final se editado), os riscos manuais, os totais
(mostrados/escolhidos por origem, tamanho da matriz, participacao
percentual, adocao da IA, editados por origem - no geral e por categoria),
as falhas de IA e as respostas do formulario final.
"""

from __future__ import annotations

import difflib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2

DEFAULT_LOG_DIR = Path("data/avaliacao")

ORIGIN_CATALOG = "catalogo"
ORIGIN_AI = "ia"
ORIGIN_MANUAL = "manual"

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class SuggestionLogEntry:
    """Um item mostrado na aba Sugestoes (catalogo ou IA)."""

    slot_id: str
    categoria: str
    origem: str  # ORIGIN_CATALOG | ORIGIN_AI - nunca exibido na interface
    id_catalogo: str | None  # so para origem == ORIGIN_CATALOG
    posicao: int  # posicao sorteada dentro da categoria (1, 2, ...)
    escolhido: bool
    editado: bool = False
    proporcao_edicao: float = 0.0
    texto_mostrado: dict = field(default_factory=dict)
    texto_final: dict | None = None  # so quando editado


@dataclass
class ManualRiskLogEntry:
    """Um risco incluido manualmente na revisao."""

    id: str
    categoria: str
    escolhido: bool
    texto: dict = field(default_factory=dict)


@dataclass
class CategoryTotals:
    categoria: str
    mostrados_catalogo: int = 0
    mostrados_ia: int = 0
    escolhidos_catalogo: int = 0
    escolhidos_ia: int = 0
    escolhidos_manual: int = 0


@dataclass
class EvaluationTotals:
    mostrados_catalogo: int = 0
    mostrados_ia: int = 0
    escolhidos_catalogo: int = 0
    escolhidos_ia: int = 0
    escolhidos_manual: int = 0
    tamanho_matriz_final: int = 0
    participacao_catalogo_pct: float = 0.0
    participacao_ia_pct: float = 0.0
    participacao_manual_pct: float = 0.0
    adocao_ia_pct: float = 0.0  # escolhidos_ia / mostrados_ia * 100
    editados_catalogo: int = 0
    editados_ia: int = 0
    por_categoria: list[CategoryTotals] = field(default_factory=list)


@dataclass
class EvaluationLog:
    codigo_processo: str
    idioma: str
    provedor_ia: str
    modelo_ia: str
    versao_prompt: str
    hash_biblioteca: str
    semente: int
    versao_schema: int = SCHEMA_VERSION
    nome_caso: str | None = None
    experiencia: str = ""
    data_hora: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sugestoes: list[SuggestionLogEntry] = field(default_factory=list)
    riscos_manuais: list[ManualRiskLogEntry] = field(default_factory=list)
    totais: EvaluationTotals = field(default_factory=EvaluationTotals)
    falhas_ia: list[dict] = field(default_factory=list)
    respostas_finais: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def edit_ratio(original: str, editado: str) -> float:
    """Proporcao de edicao via difflib: 0 = identico, 1 = totalmente diferente."""
    original = original or ""
    editado = editado or ""
    if not original and not editado:
        return 0.0
    matcher = difflib.SequenceMatcher(a=original, b=editado)
    return round(1.0 - matcher.ratio(), 4)


def compute_totals(
    items_by_slot: dict[str, dict],
    review_decisions: dict[str, dict],
    selected_rows: list,
    slot_origin: dict[str, str],
) -> EvaluationTotals:
    """Totais da avaliacao, no geral e por categoria.

    ``items_by_slot`` (aba Sugestoes) da os "mostrados" por origem;
    ``selected_rows`` (o que de fato entrou na matriz final, depois da
    Revisao - inclui a segunda caixa "Incluir na matriz" do editor) da os
    "escolhidos"/a composicao final, por origem (via ``slot_origin``);
    ``review_decisions`` (Parte 2) da os editados, so para os escolhidos.
    """

    def origin_of(row) -> str:
        return slot_origin.get(row.id, ORIGIN_MANUAL)

    mostrados_catalogo = sum(1 for info in items_by_slot.values() if info["origin"] == ORIGIN_CATALOG)
    mostrados_ia = sum(1 for info in items_by_slot.values() if info["origin"] == ORIGIN_AI)

    escolhidos_catalogo = sum(1 for row in selected_rows if origin_of(row) == ORIGIN_CATALOG)
    escolhidos_ia = sum(1 for row in selected_rows if origin_of(row) == ORIGIN_AI)
    escolhidos_manual = sum(1 for row in selected_rows if origin_of(row) == ORIGIN_MANUAL)
    tamanho_matriz = len(selected_rows)

    def pct(count: int) -> float:
        return round(count / tamanho_matriz * 100, 1) if tamanho_matriz else 0.0

    editados_catalogo = sum(
        1
        for slot, decision in review_decisions.items()
        if decision.get("editado") and slot_origin.get(slot) == ORIGIN_CATALOG
    )
    editados_ia = sum(
        1
        for slot, decision in review_decisions.items()
        if decision.get("editado") and slot_origin.get(slot) == ORIGIN_AI
    )

    categorias = sorted({info["categoria"] for info in items_by_slot.values()} | {row.categoria for row in selected_rows})
    por_categoria = []
    for categoria in categorias:
        slots_in_category = {slot: info for slot, info in items_by_slot.items() if info["categoria"] == categoria}
        rows_in_category = [row for row in selected_rows if row.categoria == categoria]
        por_categoria.append(
            CategoryTotals(
                categoria=categoria,
                mostrados_catalogo=sum(1 for info in slots_in_category.values() if info["origin"] == ORIGIN_CATALOG),
                mostrados_ia=sum(1 for info in slots_in_category.values() if info["origin"] == ORIGIN_AI),
                escolhidos_catalogo=sum(1 for row in rows_in_category if origin_of(row) == ORIGIN_CATALOG),
                escolhidos_ia=sum(1 for row in rows_in_category if origin_of(row) == ORIGIN_AI),
                escolhidos_manual=sum(1 for row in rows_in_category if origin_of(row) == ORIGIN_MANUAL),
            )
        )

    return EvaluationTotals(
        mostrados_catalogo=mostrados_catalogo,
        mostrados_ia=mostrados_ia,
        escolhidos_catalogo=escolhidos_catalogo,
        escolhidos_ia=escolhidos_ia,
        escolhidos_manual=escolhidos_manual,
        tamanho_matriz_final=tamanho_matriz,
        participacao_catalogo_pct=pct(escolhidos_catalogo),
        participacao_ia_pct=pct(escolhidos_ia),
        participacao_manual_pct=pct(escolhidos_manual),
        adocao_ia_pct=round(escolhidos_ia / mostrados_ia * 100, 1) if mostrados_ia else 0.0,
        editados_catalogo=editados_catalogo,
        editados_ia=editados_ia,
        por_categoria=por_categoria,
    )


def _safe_filename(codigo_processo: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", codigo_processo.strip()).strip("_")
    return (cleaned or "processo")[:80]


def save_evaluation_log(
    log: EvaluationLog, directory: str | Path = DEFAULT_LOG_DIR, session_id: str | None = None
) -> Path:
    """Grava ``log`` como um JSON em ``directory`` (criado se necessario).

    Com ``session_id``, o nome do arquivo e deterministico
    (``<codigo>_<session_id>.json``): salvar de novo o mesmo processo na
    mesma sessao sobrescreve o arquivo, sem criar outro. Sem ``session_id``,
    usa timestamp + um sufixo aleatorio curto (o timestamp sozinho pode
    colidir em chamadas seguidas - a resolucao do relogio varia por SO).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = session_id or f"{timestamp}-{uuid.uuid4().hex[:6]}"
    path = directory / f"{_safe_filename(log.codigo_processo)}_{suffix}.json"
    path.write_text(json.dumps(log.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
