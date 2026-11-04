"""Metadados registrados em toda execucao: versoes, modelo, parametros, sementes.

Cada relatorio inclui isso para que uma execucao possa ser auditada e, na
medida do possivel, refeita.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata

from .config import RunConfig

TRACKED_PACKAGES = ("requests", "sentence-transformers", "torch", "pytest")


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


@dataclass(frozen=True)
class RunMetadata:
    python_version: str
    platform: str
    package_versions: dict[str, str]
    app_git_commit: str
    config: dict

    def to_dict(self) -> dict:
        return asdict(self)


def collect_run_metadata(config: RunConfig) -> RunMetadata:
    return RunMetadata(
        python_version=sys.version,
        platform=platform.platform(),
        package_versions=_package_versions(),
        app_git_commit=_git_commit(),
        config=_config_to_dict(config),
    )


def _config_to_dict(config: RunConfig) -> dict:
    payload = asdict(config)
    payload["engines"] = list(config.engines)
    payload["k_values"] = list(config.k_values)
    return payload
