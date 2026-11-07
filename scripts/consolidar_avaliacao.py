"""Consolida os logs de avaliacao cega (``data/avaliacao/*.json``).

Gera ``resumo_processos.csv`` (uma linha por processo) e ``resumo_pares.csv``,
e imprime no terminal os totais, as taxas por origem e o teste de McNemar
exato entre ``so_biblioteca`` e ``so_ia``.

Uso:
    python scripts/consolidar_avaliacao.py [--log-dir data/avaliacao] [--out-dir .]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_log import (  # noqa: E402
    OUTCOME_AI_ONLY,
    OUTCOME_BOTH,
    OUTCOME_LIBRARY_ONLY,
    OUTCOME_NONE,
    OUTCOME_SEM_PAR,
)

DEFAULT_LOG_DIR = ROOT / "data" / "avaliacao"
OUTCOME_ORDER = [OUTCOME_LIBRARY_ONLY, OUTCOME_AI_ONLY, OUTCOME_BOTH, OUTCOME_NONE, OUTCOME_SEM_PAR]

PROCESS_SUMMARY_FIELDS = [
    "codigo_processo",
    "data_hora",
    "idioma",
    "provedor_ia",
    "modelo_ia",
    "versao_prompt",
    "hash_biblioteca",
    "semente",
    "total_sugestoes",
    "total_pares",
    "escolhidos_biblioteca",
    "escolhidos_ia",
    "riscos_manuais",
    "editados",
    "so_biblioteca",
    "so_ia",
    "ambos",
    "nenhum",
    "sem_par",
    "utilidade",
    "confianca",
    "esforco",
    "comentario",
]

PAIR_SUMMARY_FIELDS = ["codigo_processo", "par", "categoria", "desfecho"]


def load_logs(log_dir: Path) -> list[dict]:
    logs = []
    for path in sorted(Path(log_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            logs.append(json.load(handle))
    return logs


def process_summary_row(log: dict) -> dict:
    sugestoes = log.get("sugestoes", [])
    pares = log.get("pares", [])
    outcome_counts = Counter(pair.get("desfecho") for pair in pares)
    respostas = log.get("respostas_finais") or {}
    return {
        "codigo_processo": log.get("codigo_processo", ""),
        "data_hora": log.get("data_hora", ""),
        "idioma": log.get("idioma", ""),
        "provedor_ia": log.get("provedor_ia", ""),
        "modelo_ia": log.get("modelo_ia", ""),
        "versao_prompt": log.get("versao_prompt", ""),
        "hash_biblioteca": log.get("hash_biblioteca", ""),
        "semente": log.get("semente", ""),
        "total_sugestoes": len(sugestoes),
        "total_pares": len(pares),
        "escolhidos_biblioteca": sum(
            1 for s in sugestoes if s.get("origem") == "biblioteca" and s.get("escolhido")
        ),
        "escolhidos_ia": sum(1 for s in sugestoes if s.get("origem") == "ia" and s.get("escolhido")),
        "riscos_manuais": log.get("quantidade_riscos_manuais", 0),
        "editados": sum(1 for s in sugestoes if s.get("editado")),
        "so_biblioteca": outcome_counts.get(OUTCOME_LIBRARY_ONLY, 0),
        "so_ia": outcome_counts.get(OUTCOME_AI_ONLY, 0),
        "ambos": outcome_counts.get(OUTCOME_BOTH, 0),
        "nenhum": outcome_counts.get(OUTCOME_NONE, 0),
        "sem_par": outcome_counts.get(OUTCOME_SEM_PAR, 0),
        "utilidade": respostas.get("utilidade", ""),
        "confianca": respostas.get("confianca", ""),
        "esforco": respostas.get("esforco", ""),
        "comentario": respostas.get("comentario", ""),
    }


def pair_summary_rows(log: dict) -> list[dict]:
    codigo = log.get("codigo_processo", "")
    return [
        {
            "codigo_processo": codigo,
            "par": pair.get("par"),
            "categoria": pair.get("categoria"),
            "desfecho": pair.get("desfecho"),
        }
        for pair in log.get("pares", [])
    ]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Teste exato de McNemar (bicaudal), sem dependencias externas.

    ``b``/``c`` sao as contagens discordantes (so_biblioteca / so_ia); os
    pares concordantes (ambos ou nenhum escolhidos) nao entram no teste.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cumulative = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * cumulative)


def print_report(process_rows: list[dict]) -> None:
    total_processos = len(process_rows)
    total_pares = sum(row["total_pares"] for row in process_rows)
    outcome_totals = Counter()
    for row in process_rows:
        for key in ["so_biblioteca", "so_ia", "ambos", "nenhum", "sem_par"]:
            outcome_totals[key] += row[key]

    print(f"Processos: {total_processos}")
    print(f"Pares avaliados: {total_pares}")
    print()
    print("Desfechos por par:")
    for key in ["so_biblioteca", "so_ia", "ambos", "nenhum", "sem_par"]:
        count = outcome_totals[key]
        rate = (count / total_pares * 100) if total_pares else 0.0
        print(f"  {key}: {count} ({rate:.1f}%)")
    print()

    total_sugestoes = sum(row["total_sugestoes"] for row in process_rows)
    escolhidos_biblioteca = sum(row["escolhidos_biblioteca"] for row in process_rows)
    escolhidos_ia = sum(row["escolhidos_ia"] for row in process_rows)
    print(f"Sugestoes no total: {total_sugestoes}")
    print(f"  Escolhidas da biblioteca: {escolhidos_biblioteca}")
    print(f"  Escolhidas da IA: {escolhidos_ia}")
    print()

    b = outcome_totals["so_biblioteca"]
    c = outcome_totals["so_ia"]
    p_value = mcnemar_exact_pvalue(b, c)
    print("Teste de McNemar (so_biblioteca x so_ia, exato, bicaudal):")
    print(f"  so_biblioteca (b) = {b}, so_ia (c) = {c}, n = {b + c}")
    print(f"  p-valor = {p_value:.4f}")


def consolidate(log_dir: Path, out_dir: Path) -> tuple[list[dict], list[dict]]:
    logs = load_logs(log_dir)
    process_rows = [process_summary_row(log) for log in logs]
    pair_rows = [row for log in logs for row in pair_summary_rows(log)]
    if logs:
        write_csv(Path(out_dir) / "resumo_processos.csv", PROCESS_SUMMARY_FIELDS, process_rows)
        write_csv(Path(out_dir) / "resumo_pares.csv", PAIR_SUMMARY_FIELDS, pair_rows)
    return process_rows, pair_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Diretorio com os JSON de avaliacao.")
    parser.add_argument("--out-dir", type=Path, default=ROOT, help="Diretorio onde gravar os CSVs de resumo.")
    args = parser.parse_args(argv)

    process_rows, _ = consolidate(args.log_dir, args.out_dir)
    if not process_rows:
        print(f"Nenhum log encontrado em {args.log_dir}.")
        return

    print_report(process_rows)


if __name__ == "__main__":
    main()
