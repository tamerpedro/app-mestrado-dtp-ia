"""Consolida os logs de avaliacao com catalogo+IA (``data/avaliacao/*.json``, schema v2).

So estatistica descritiva: quanto do processo com catalogo e IA foi adotado,
qual a participacao de cada origem na matriz final e como isso varia com a
experiencia de quem elaborou a matriz - nao ha teste de hipotese nem
comparacao direta entre a taxa da IA e a do catalogo (nao e uma competicao
entre as duas origens).

Gera ``resumo_processos.csv`` (uma linha por processo) e
``resumo_categorias.csv`` (composicao da matriz final por categoria, somada
entre todos os processos), e imprime no terminal um relatorio geral e um por
faixa de experiencia. Logs de schema antigo (pareamento, anterior a esta
versao) sao pulados com um aviso.

Uso:
    python scripts/consolidar_avaliacao.py [--log-dir data/avaliacao] [--out-dir .]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_log import log_consistency_errors  # noqa: E402

SCHEMA_VERSION = 2
DEFAULT_LOG_DIR = ROOT / "data" / "avaliacao"

EXPERIENCE_ORDER = ["ate_2", "2_a_5", "mais_5"]
EXPERIENCE_UNKNOWN = "nao_informado"
EXPERIENCE_LABELS = {
    "ate_2": "Ate 2 anos",
    "2_a_5": "De 2 a 5 anos",
    "mais_5": "Mais de 5 anos",
    EXPERIENCE_UNKNOWN: "Nao informado",
}

PROCESS_SUMMARY_FIELDS = [
    "codigo_processo",
    "nome_caso",
    "experiencia",
    "idioma",
    "provedor_ia",
    "modelo_ia",
    "versao_prompt",
    "data_hora",
    "mostrados_catalogo",
    "mostrados_ia",
    "escolhidos_catalogo",
    "escolhidos_ia",
    "escolhidos_manual",
    "tamanho_matriz_final",
    "participacao_catalogo_pct",
    "participacao_ia_pct",
    "participacao_manual_pct",
    "adocao_ia_pct",
    "editados_catalogo",
    "editados_ia",
    "avaliacao_alterada_catalogo",
    "avaliacao_alterada_ia",
    "desmarcados_na_revisao_catalogo",
    "desmarcados_na_revisao_ia",
    "desmarcados_na_revisao_manual",
    "categoria_alterada_catalogo",
    "categoria_alterada_ia",
    "categoria_alterada_manual",
    "riscos_manuais",
    "utilidade",
    "confianca",
    "esforco",
    "comentario",
]

CATEGORY_SUMMARY_FIELDS = [
    "categoria",
    "escolhidos_catalogo",
    "escolhidos_ia",
    "escolhidos_manual",
    "total",
    "participacao_catalogo_pct",
    "participacao_ia_pct",
    "participacao_manual_pct",
]


def load_logs(log_dir: Path) -> list[dict]:
    """Logs de schema v2 em ``log_dir``; schema antigo (ou ausente) e pulado
    com um aviso no terminal."""
    logs = []
    for path in sorted(Path(log_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            log = json.load(handle)
        versao = log.get("versao_schema")
        if versao != SCHEMA_VERSION:
            print(f"Aviso: {path.name} usa schema antigo (versao_schema={versao!r}); pulando.")
            continue
        errors = log_consistency_errors(log)
        if errors:
            print(f"AVISO: {path.name} falhou na verificacao de consistencia:")
            for error in errors:
                print(f"  - {error}")
        logs.append(log)
    return logs


def experience_band(log: dict) -> str:
    experiencia = (log.get("experiencia") or "").strip()
    return experiencia if experiencia in EXPERIENCE_ORDER else EXPERIENCE_UNKNOWN


def _desmarcados_na_revisao(sugestoes: list[dict], riscos_manuais: list[dict], origem: str) -> int:
    """Itens que avancaram ate a Revisao mas nao ficaram na matriz final.

    Catalogo/IA: escolhidos na aba Sugestoes e depois desmarcados em
    "Incluir na matriz". Manuais: criados na Revisao (ja nascem marcados)
    e depois desmarcados - nao ha uma etapa de "Sugestoes" para eles.
    """
    if origem == "manual":
        return sum(1 for item in riscos_manuais if not item.get("escolhido"))
    return sum(
        1
        for item in sugestoes
        if item.get("origem") == origem and item.get("escolhido") and not item.get("incluido_matriz")
    )


def _categoria_alterada(sugestoes: list[dict], riscos_manuais: list[dict], origem: str) -> int:
    if origem == "manual":
        return sum(1 for item in riscos_manuais if item.get("categoria_alterada"))
    return sum(1 for item in sugestoes if item.get("origem") == origem and item.get("categoria_alterada"))


def process_summary_row(log: dict) -> dict:
    totais = log.get("totais") or {}
    respostas = log.get("respostas_finais") or {}
    sugestoes = log.get("sugestoes") or []
    riscos_manuais = log.get("riscos_manuais") or []
    return {
        "codigo_processo": log.get("codigo_processo", ""),
        "nome_caso": log.get("nome_caso") or "",
        "experiencia": log.get("experiencia") or "",
        "idioma": log.get("idioma", ""),
        "provedor_ia": log.get("provedor_ia", ""),
        "modelo_ia": log.get("modelo_ia", ""),
        "versao_prompt": log.get("versao_prompt", ""),
        "data_hora": log.get("data_hora", ""),
        "mostrados_catalogo": totais.get("mostrados_catalogo", 0),
        "mostrados_ia": totais.get("mostrados_ia", 0),
        "escolhidos_catalogo": totais.get("escolhidos_catalogo", 0),
        "escolhidos_ia": totais.get("escolhidos_ia", 0),
        "escolhidos_manual": totais.get("escolhidos_manual", 0),
        "tamanho_matriz_final": totais.get("tamanho_matriz_final", 0),
        "participacao_catalogo_pct": totais.get("participacao_catalogo_pct", 0.0),
        "participacao_ia_pct": totais.get("participacao_ia_pct", 0.0),
        "participacao_manual_pct": totais.get("participacao_manual_pct", 0.0),
        "adocao_ia_pct": totais.get("adocao_ia_pct", 0.0),
        "editados_catalogo": totais.get("editados_catalogo", 0),
        "editados_ia": totais.get("editados_ia", 0),
        "avaliacao_alterada_catalogo": totais.get("avaliacao_alterada_catalogo", 0),
        "avaliacao_alterada_ia": totais.get("avaliacao_alterada_ia", 0),
        "desmarcados_na_revisao_catalogo": _desmarcados_na_revisao(sugestoes, riscos_manuais, "catalogo"),
        "desmarcados_na_revisao_ia": _desmarcados_na_revisao(sugestoes, riscos_manuais, "ia"),
        "desmarcados_na_revisao_manual": _desmarcados_na_revisao(sugestoes, riscos_manuais, "manual"),
        "categoria_alterada_catalogo": _categoria_alterada(sugestoes, riscos_manuais, "catalogo"),
        "categoria_alterada_ia": _categoria_alterada(sugestoes, riscos_manuais, "ia"),
        "categoria_alterada_manual": _categoria_alterada(sugestoes, riscos_manuais, "manual"),
        "riscos_manuais": len(riscos_manuais),
        "utilidade": respostas.get("utilidade", ""),
        "confianca": respostas.get("confianca", ""),
        "esforco": respostas.get("esforco", ""),
        "comentario": respostas.get("comentario", ""),
    }


def category_summary_rows(logs: list[dict]) -> list[dict]:
    """Composicao da matriz final por categoria, somada entre todos os processos."""
    totals_by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"catalogo": 0, "ia": 0, "manual": 0})
    for log in logs:
        for categoria_totais in (log.get("totais") or {}).get("por_categoria", []):
            categoria = categoria_totais.get("categoria", "")
            bucket = totals_by_category[categoria]
            bucket["catalogo"] += categoria_totais.get("escolhidos_catalogo", 0)
            bucket["ia"] += categoria_totais.get("escolhidos_ia", 0)
            bucket["manual"] += categoria_totais.get("escolhidos_manual", 0)

    rows = []
    for categoria in sorted(totals_by_category):
        bucket = totals_by_category[categoria]
        total = bucket["catalogo"] + bucket["ia"] + bucket["manual"]

        def pct(count: int) -> float:
            return round(count / total * 100, 1) if total else 0.0

        rows.append(
            {
                "categoria": categoria,
                "escolhidos_catalogo": bucket["catalogo"],
                "escolhidos_ia": bucket["ia"],
                "escolhidos_manual": bucket["manual"],
                "total": total,
                "participacao_catalogo_pct": pct(bucket["catalogo"]),
                "participacao_ia_pct": pct(bucket["ia"]),
                "participacao_manual_pct": pct(bucket["manual"]),
            }
        )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _print_group_report(label: str, rows: list[dict]) -> None:
    print(f"{label}: {len(rows)} processo(s)")
    if not rows:
        return
    print(f"  Tamanho medio da matriz final: {_mean([row['tamanho_matriz_final'] for row in rows])}")
    print(f"  Participacao media do catalogo: {_mean([row['participacao_catalogo_pct'] for row in rows])}%")
    print(f"  Participacao media da IA: {_mean([row['participacao_ia_pct'] for row in rows])}%")
    print(f"  Participacao media manual: {_mean([row['participacao_manual_pct'] for row in rows])}%")
    print(f"  Adocao media das sugestoes de IA: {_mean([row['adocao_ia_pct'] for row in rows])}%")
    print(f"  Media de riscos manuais: {_mean([row['riscos_manuais'] for row in rows])}")
    utilidade = [row["utilidade"] for row in rows if isinstance(row["utilidade"], (int, float))]
    confianca = [row["confianca"] for row in rows if isinstance(row["confianca"], (int, float))]
    esforco = [row["esforco"] for row in rows if isinstance(row["esforco"], (int, float))]
    print(f"  Utilidade media: {_mean(utilidade)}")
    print(f"  Confianca media: {_mean(confianca)}")
    print(f"  Esforco medio: {_mean(esforco)}")


def print_report(process_rows: list[dict]) -> None:
    print(f"Processos: {len(process_rows)}")
    print()
    _print_group_report("Geral", process_rows)
    print()

    by_band: dict[str, list[dict]] = defaultdict(list)
    for row in process_rows:
        experiencia = row["experiencia"] if row["experiencia"] in EXPERIENCE_ORDER else EXPERIENCE_UNKNOWN
        by_band[experiencia].append(row)

    for band in [*EXPERIENCE_ORDER, EXPERIENCE_UNKNOWN]:
        rows = by_band.get(band, [])
        if not rows:
            continue
        _print_group_report(f"Experiencia: {EXPERIENCE_LABELS[band]}", rows)
        print()


def consolidate(log_dir: Path, out_dir: Path) -> tuple[list[dict], list[dict]]:
    logs = load_logs(log_dir)
    process_rows = [process_summary_row(log) for log in logs]
    category_rows = category_summary_rows(logs)
    if logs:
        write_csv(Path(out_dir) / "resumo_processos.csv", PROCESS_SUMMARY_FIELDS, process_rows)
        write_csv(Path(out_dir) / "resumo_categorias.csv", CATEGORY_SUMMARY_FIELDS, category_rows)
    return process_rows, category_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Diretorio com os JSON de avaliacao.")
    parser.add_argument("--out-dir", type=Path, default=ROOT, help="Diretorio onde gravar os CSVs de resumo.")
    args = parser.parse_args(argv)

    process_rows, _ = consolidate(args.log_dir, args.out_dir)
    if not process_rows:
        print(f"Nenhum log de schema v{SCHEMA_VERSION} encontrado em {args.log_dir}.")
        return

    print_report(process_rows)


if __name__ == "__main__":
    main()
