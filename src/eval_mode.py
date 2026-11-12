"""Modo de avaliacao cega: biblioteca congelada durante o segundo ciclo DSR.

Com ``MODO_AVALIACAO`` ligado (padrao nesta branch), nada e gravado em
``data/riscos_base.csv``: riscos de IA escolhidos vao para
``data/candidatos_ia.csv`` e o botao de salvar na biblioteca da revisao
humana fica desativado (ver ``src.risk_library``).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

try:
    import streamlit as st
except ImportError:  # pragma: no cover - streamlit sempre disponivel em producao
    st = None  # type: ignore[assignment]


DEFAULT_MODO_AVALIACAO = True


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _secret(key: str) -> str | None:
    if st is None:
        return None
    try:
        value = st.secrets.get(key)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - secrets.toml ausente ou invalido
        return None
    return str(value) if value is not None else None


def is_modo_avaliacao() -> bool:
    """Le ``MODO_AVALIACAO`` de env ou ``secrets.toml``; padrao ligado."""
    parsed = _parse_bool(os.environ.get("MODO_AVALIACAO"))
    if parsed is not None:
        return parsed
    parsed = _parse_bool(_secret("MODO_AVALIACAO"))
    if parsed is not None:
        return parsed
    return DEFAULT_MODO_AVALIACAO


def library_hash(path: str | Path) -> str:
    """Hash SHA-256 (hex) do conteudo bruto do CSV da biblioteca de riscos."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
