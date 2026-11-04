"""Entrypoint de linha de comando: ``python -m evaluation``.

Roda a avaliacao sobre os casos reais em ``evaluation/cases/real/`` (vazio
em um checkout limpo -- ver ``evaluation/README.md`` para como adicionar os
seus) e grava o relatorio em ``evaluation/reports/<timestamp>/``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .config import EmbeddingConfig, LLMConfig, RunConfig
from .runner import run_evaluation, write_report

DEFAULT_RISKS_PATH = Path(__file__).resolve().parent.parent / "data" / "riscos_base.csv"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avaliacao comparativa dos motores de sugestao de riscos.")
    parser.add_argument("--risks", default=str(DEFAULT_RISKS_PATH), help="Caminho para riscos_base.csv")
    parser.add_argument("--output", default=None, help="Diretorio de saida do relatorio")
    parser.add_argument("--seed", type=int, default=RunConfig().seed)
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Inclui os casos sinteticos de exemplo na avaliacao (uteis so para demonstracao).",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "hashing"],
        default="sentence-transformers",
        help="'hashing' e um fallback local sem modelo real -- nao usar para os numeros finais.",
    )
    parser.add_argument("--enable-llm", action="store_true", help="Habilita o motor LLM (desligado por padrao).")
    parser.add_argument("--llm-endpoint", default=RunConfig().llm.endpoint)
    parser.add_argument("--llm-model", default=RunConfig().llm.model)
    parser.add_argument(
        "--llm-allow-external",
        action="store_true",
        help="Permite um endpoint LLM fora da maquina local. Use com cautela.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    config = RunConfig(
        seed=args.seed,
        embedding=EmbeddingConfig(backend=args.embedding_backend),
        llm=LLMConfig(
            enabled=args.enable_llm,
            endpoint=args.llm_endpoint,
            model=args.llm_model,
            allow_external=args.llm_allow_external,
        ),
    )

    report = run_evaluation(
        risks_path=args.risks,
        config=config,
        include_real_cases=True,
        include_synthetic_cases=args.include_synthetic,
    )

    if not report.records:
        print(
            "Nenhum caso disponivel para avaliacao principal. Adicione casos em "
            "evaluation/cases/real/ (nao versionados) ou use --include-synthetic "
            "para uma demonstracao com o caso de exemplo."
        )
        return 1

    output_dir = Path(args.output) if args.output else DEFAULT_REPORTS_DIR / datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H%M%SZ"
    )
    written_to = write_report(report, output_dir)
    print(f"Relatorio gravado em {written_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
