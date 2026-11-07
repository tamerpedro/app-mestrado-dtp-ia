"""Consolida os logs de avaliacao cega (``data/avaliacao/*.json``).

Gera ``resumo_processos.csv`` (uma linha por processo) e ``resumo_pares.csv``
(uma linha por par, com o grupo de corte da sugestao de biblioteca), e
imprime no terminal os totais e as taxas por desfecho separados para pares
ACIMA do corte, ABAIXO do corte e ``sem_par`` (sem risco de biblioteca). O
teste de McNemar exato entre ``so_biblioteca`` e ``so_ia`` usa so os pares
ACIMA do corte - pares abaixo do corte ou sem_par nao representam uma
comparacao justa entre as duas origens.

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

# Desfechos do schema antigo (pareamento 1 biblioteca + 1 IA por categoria).
# src.eval_log nao tem mais esses nomes (o pareamento foi substituido pela
# lista cega catalogo+IA, ver src/catalog_mix.py); este script sera reescrito
# para o novo schema numa proxima etapa. Ate la, so processa logs antigos.
OUTCOME_LIBRARY_ONLY = "so_biblioteca"
OUTCOME_AI_ONLY = "so_ia"
OUTCOME_BOTH = "ambos"
OUTCOME_NONE = "nenhum"
OUTCOME_SEM_PAR = "sem_par"

DEFAULT_LOG_DIR = ROOT / "data" / "avaliacao"
OUTCOME_ORDER = [OUTCOME_LIBRARY_ONLY, OUTCOME_AI_ONLY, OUTCOME_BOTH, OUTCOME_NONE, OUTCOME_SEM_PAR]

# Grupo de corte do par, pela sugestao de biblioteca (nunca exposto na interface).
GROUP_ACIMA_CORTE = "acima_corte"
GROUP_ABAIXO_CORTE = "abaixo_corte"
GROUP_SEM_PAR = "sem_par"
GROUP_LABELS = {
    GROUP_ACIMA_CORTE: "Pares acima do corte",
    GROUP_ABAIXO_CORTE: "Pares abaixo do corte",
    GROUP_SEM_PAR: "Pares sem risco de biblioteca (sem_par)",
}

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
    "pares_acima_corte",
    "pares_abaixo_corte",
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

PAIR_SUMMARY_FIELDS = ["codigo_processo", "par", "categoria", "desfecho", "grupo_corte", "pontuacao_biblioteca"]


def pair_cutoff_group(log: dict, pair: dict) -> str:
    """Classifica o par pelo corte da sua sugestao de biblioteca.

    ``sem_par`` quando a categoria nao tem nenhum risco de biblioteca;
    ``abaixo_corte``/``acima_corte`` conforme o campo ``abaixo_corte`` que
    ``src.eval_log.SuggestionLogEntry`` grava para a sugestao de biblioteca
    do par. Nunca exposto na interface, so para consolidacao.
    """
    if pair.get("desfecho") == OUTCOME_SEM_PAR:
        return GROUP_SEM_PAR
    biblioteca = next(
        (
            s
            for s in log.get("sugestoes", [])
            if s.get("par") == pair.get("par") and s.get("origem") == "biblioteca"
        ),
        None,
    )
    if biblioteca is None:
        return GROUP_SEM_PAR
    return GROUP_ABAIXO_CORTE if biblioteca.get("abaixo_corte") else GROUP_ACIMA_CORTE


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
    group_counts = Counter(pair_cutoff_group(log, pair) for pair in pares)
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
        "pares_acima_corte": group_counts.get(GROUP_ACIMA_CORTE, 0),
        "pares_abaixo_corte": group_counts.get(GROUP_ABAIXO_CORTE, 0),
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
    rows = []
    for pair in log.get("pares", []):
        biblioteca = next(
            (
                s
                for s in log.get("sugestoes", [])
                if s.get("par") == pair.get("par") and s.get("origem") == "biblioteca"
            ),
            None,
        )
        rows.append(
            {
                "codigo_processo": codigo,
                "par": pair.get("par"),
                "categoria": pair.get("categoria"),
                "desfecho": pair.get("desfecho"),
                "grupo_corte": pair_cutoff_group(log, pair),
                "pontuacao_biblioteca": biblioteca.get("pontuacao") if biblioteca else "",
            }
        )
    return rows


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


def _print_outcomes_for_group(pair_rows: list[dict], group: str) -> None:
    group_rows = [row for row in pair_rows if row["grupo_corte"] == group]
    print(f"{GROUP_LABELS[group]}: {len(group_rows)}")
    if not group_rows:
        return
    outcome_counts = Counter(row["desfecho"] for row in group_rows)
    for code in OUTCOME_ORDER:
        count = outcome_counts.get(code, 0)
        if count:
            rate = count / len(group_rows) * 100
            print(f"  {code}: {count} ({rate:.1f}%)")


def print_report(process_rows: list[dict], pair_rows: list[dict]) -> None:
    print(f"Processos: {len(process_rows)}")
    print(f"Pares avaliados: {len(pair_rows)}")
    print()

    for group in [GROUP_ACIMA_CORTE, GROUP_ABAIXO_CORTE, GROUP_SEM_PAR]:
        _print_outcomes_for_group(pair_rows, group)
        print()

    total_sugestoes = sum(row["total_sugestoes"] for row in process_rows)
    escolhidos_biblioteca = sum(row["escolhidos_biblioteca"] for row in process_rows)
    escolhidos_ia = sum(row["escolhidos_ia"] for row in process_rows)
    print(f"Sugestoes no total: {total_sugestoes}")
    print(f"  Escolhidas da biblioteca: {escolhidos_biblioteca}")
    print(f"  Escolhidas da IA: {escolhidos_ia}")
    print()

    acima_rows = [row for row in pair_rows if row["grupo_corte"] == GROUP_ACIMA_CORTE]
    outcome_counts_acima = Counter(row["desfecho"] for row in acima_rows)
    b = outcome_counts_acima.get(OUTCOME_LIBRARY_ONLY, 0)
    c = outcome_counts_acima.get(OUTCOME_AI_ONLY, 0)
    p_value = mcnemar_exact_pvalue(b, c)
    print("Teste de McNemar (so_biblioteca x so_ia, exato, bicaudal, so pares acima do corte):")
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

    process_rows, pair_rows = consolidate(args.log_dir, args.out_dir)
    if not process_rows:
        print(f"Nenhum log encontrado em {args.log_dir}.")
        return

    print_report(process_rows, pair_rows)


if __name__ == "__main__":
    main()
