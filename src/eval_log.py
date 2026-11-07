"""Schema e gravacao do log de avaliacao, um JSON por processo.

Grava em ``data/avaliacao/`` (fora do controle de versao): metadados do
processo (idioma, provedor/modelo de IA, versao do prompt, hash da
biblioteca, semente do sorteio) e o detalhe de cada item mostrado
(categoria, origem, posicao, se foi escolhido/editado e a proporcao de
edicao via ``difflib``). O schema completo (totais por origem/categoria,
experiencia do participante etc.) e montado numa proxima etapa; por
enquanto so o essencial para nao perder o formato ja pensado.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_DIR = Path("data/avaliacao")

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class SuggestionLogEntry:
    categoria: str
    origem: str  # "catalogo" | "ia" - nunca exibido na interface
    posicao: int  # posicao sorteada dentro da categoria (1, 2, ...)
    slot_id: str
    escolhido: bool
    editado: bool = False
    proporcao_edicao: float = 0.0


@dataclass
class EvaluationLog:
    codigo_processo: str
    idioma: str
    provedor_ia: str
    modelo_ia: str
    versao_prompt: str
    hash_biblioteca: str
    semente: int
    data_hora: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sugestoes: list[SuggestionLogEntry] = field(default_factory=list)
    quantidade_riscos_manuais: int = 0
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


def _safe_filename(codigo_processo: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", codigo_processo.strip()).strip("_")
    return (cleaned or "processo")[:80]


def save_evaluation_log(log: EvaluationLog, directory: str | Path = DEFAULT_LOG_DIR) -> Path:
    """Grava ``log`` como um JSON em ``directory`` (criado se necessario)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{_safe_filename(log.codigo_processo)}_{timestamp}.json"
    path.write_text(json.dumps(log.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
