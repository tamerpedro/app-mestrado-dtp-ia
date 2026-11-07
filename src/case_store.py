"""Casos compartilhados: contexto da contratacao e cache de riscos de IA por caso.

Um caso e um JSON em ``data/casos/<arquivo>.json`` com os campos do
``ContractContext`` mais ``"nome"``. Para um caso salvo, os riscos de IA de
cada categoria sao gerados uma unica vez e cacheados em
``data/casos/<arquivo>_ia_<idioma>.json``; execucoes seguintes reusam o
cache, sem chamar a IA de novo - assim todos os participantes veem
exatamente os mesmos riscos de IA. Sem caso selecionado (contexto
preenchido manualmente), a geracao acontece a cada clique, sem cache.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .ai_provider import AIProvider
from .catalog_mix import generate_ai_risks_for_all_categories
from .models import ContractContext, RiskItem
from .risk_generator import GeneratedRisk

DEFAULT_CASES_DIR = Path("data/casos")
_IA_CACHE_MARKER = "_ia_"


def list_case_files(cases_dir: str | Path = DEFAULT_CASES_DIR) -> list[Path]:
    """Arquivos de caso (exclui os caches ``*_ia_<idioma>.json``)."""
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        return []
    return sorted(path for path in cases_dir.glob("*.json") if _IA_CACHE_MARKER not in path.stem)


def load_case(path: str | Path) -> tuple[str, ContractContext]:
    """Le um caso: devolve ``(nome, ContractContext)``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nome = data.get("nome") or Path(path).stem
    context = ContractContext(
        objeto=data.get("objeto", ""),
        tipo_contratacao=data.get("tipo_contratacao", ""),
        valor_estimado=float(data.get("valor_estimado") or 0),
        criticidade=data.get("criticidade", ""),
        prazo=data.get("prazo", ""),
        modalidade=data.get("modalidade", ""),
        contexto=data.get("contexto", ""),
    )
    return nome, context


def ai_cache_path(case_path: str | Path, lang: str, cases_dir: str | Path = DEFAULT_CASES_DIR) -> Path:
    return Path(cases_dir) / f"{Path(case_path).stem}_ia_{lang}.json"


def load_cached_ai_risks(
    case_path: str | Path, lang: str, cases_dir: str | Path = DEFAULT_CASES_DIR
) -> dict[str, list[GeneratedRisk]] | None:
    """Le o cache de riscos de IA do caso para o idioma, se existir."""
    cache_path = ai_cache_path(case_path, lang, cases_dir)
    if not cache_path.exists():
        return None
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {categoria: [GeneratedRisk(**item) for item in items] for categoria, items in raw.items()}


def save_cached_ai_risks(
    case_path: str | Path,
    lang: str,
    ai_risks_by_category: dict[str, list[GeneratedRisk]],
    cases_dir: str | Path = DEFAULT_CASES_DIR,
) -> Path:
    """Grava o cache de riscos de IA do caso para o idioma."""
    cache_path = ai_cache_path(case_path, lang, cases_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {categoria: [asdict(item) for item in items] for categoria, items in ai_risks_by_category.items()}
    cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_path


def get_or_generate_case_ai_risks(
    case_path: str | Path,
    lang: str,
    risks: list[RiskItem],
    context: ContractContext,
    provider: AIProvider,
    *,
    cases_dir: str | Path = DEFAULT_CASES_DIR,
    on_ai_failure=None,
) -> dict[str, list[GeneratedRisk]]:
    """Riscos de IA do caso: do cache, se existir; senao gera e grava o cache.

    Chamadas seguintes para o mesmo caso/idioma nao chamam ``provider``.
    """
    cached = load_cached_ai_risks(case_path, lang, cases_dir)
    if cached is not None:
        return cached
    generated = generate_ai_risks_for_all_categories(
        risks, context, provider, lang=lang, on_ai_failure=on_ai_failure
    )
    save_cached_ai_risks(case_path, lang, generated, cases_dir)
    return generated
