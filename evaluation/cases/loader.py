"""Carregamento dos casos de avaliacao a partir de arquivos JSON em disco."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import CaseFormatError, EvalCase, case_from_dict

DEFAULT_REAL_CASES_DIR = Path(__file__).parent / "real"
DEFAULT_SYNTHETIC_CASES_DIR = Path(__file__).parent / "synthetic"


def load_cases_from_dir(directory: str | Path) -> list[EvalCase]:
    directory = Path(directory)
    if not directory.exists():
        return []

    cases: list[EvalCase] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.append(case_from_dict(payload, source_path=str(path)))
    _check_unique_ids(cases)
    return cases


def load_real_cases(directory: str | Path = DEFAULT_REAL_CASES_DIR) -> list[EvalCase]:
    """Casos reais (nao versionados, ver .gitignore). Vazio em um checkout limpo."""
    return load_cases_from_dir(directory)


def load_synthetic_cases(directory: str | Path = DEFAULT_SYNTHETIC_CASES_DIR) -> list[EvalCase]:
    """Casos sinteticos versionados, usados em testes e como exemplo de formato."""
    return load_cases_from_dir(directory)


def _check_unique_ids(cases: list[EvalCase]) -> None:
    seen: dict[str, str] = {}
    for case in cases:
        if case.case_id in seen:
            raise CaseFormatError(
                f"case_id duplicado '{case.case_id}' em {case.source_path} e {seen[case.case_id]}"
            )
        seen[case.case_id] = case.source_path
