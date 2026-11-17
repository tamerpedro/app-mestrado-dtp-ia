"""Avaliacao retrospectiva, Parte 2c: amostra para conferencia do autor.

Sorteia 20% dos riscos reais ja julgados (minimo de 40, pelo menos um por
processo, semente fixa) e grava ``amostra_conferencia.csv`` SEM a resposta do
juiz e SEM a origem dos candidatos: uma linha por candidato do processo, com
``cobre_autor`` e ``observacao`` vazias. O autor marca ``sim`` so nos candidatos
que cobrem o risco; em branco conta como ``nao``. As linhas vem ordenadas por
processo, risco e categoria do candidato.

Como o ``candidato_id`` original revela a origem (``_cat_``/``_ia_``), a coluna
``candidato_id`` da amostra traz um codigo neutro (``K01``...), igual para todos
os riscos do mesmo processo; dentro de cada categoria a ordem e embaralhada. A correspondencia
fica em ``amostra_chave.csv``, que so deve ser aberto depois da conferencia.

Nunca sobrescreve uma amostra existente (ela pode ja ter marcacoes).

Uso:
    python scripts/retro_amostra.py --corpus "<pasta do corpus>"
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for _caminho in (ROOT, SCRIPTS):
    if str(_caminho) not in sys.path:
        sys.path.insert(0, str(_caminho))

import retro_conferencia as conf  # noqa: E402
import retro_gerar as gerar  # noqa: E402
import retro_julgar as julgar  # noqa: E402
from src.domain import CATEGORIES  # noqa: E402

AMOSTRA_CSV = "amostra_conferencia.csv"
CHAVE_CSV = "amostra_chave.csv"
AMOSTRA_FIELDS = [
    "processo_id",
    "risco_id",
    "risco_real",
    "candidato_id",
    "titulo_candidato",
    "causa_candidato",
    "consequencia_candidato",
    "cobre_autor",
    "observacao",
]
CHAVE_FIELDS = ["processo_id", "candidato_codigo", "candidato_id", "semente"]
SEMENTE_PADRAO = 20261117
FRACAO = 0.20
MINIMO = 40


class AmostraError(Exception):
    """Falha que impede gerar a amostra."""


def tamanho_da_amostra(julgados: int, processos: int) -> int:
    """20% dos julgados, no minimo 40 e um por processo, sem passar do total."""
    return min(julgados, max(math.ceil(FRACAO * julgados), MINIMO, processos))


def sortear(julgados: list[tuple[str, str]], semente: int) -> list[tuple[str, str]]:
    """Riscos sorteados, na ordem original: um por processo e o restante ao acaso."""
    rng = random.Random(semente)
    por_processo: dict[str, list[tuple[str, str]]] = {}
    for chave in julgados:
        por_processo.setdefault(chave[0], []).append(chave)
    escolhidos = {rng.choice(riscos) for riscos in por_processo.values()}
    restantes = [chave for chave in julgados if chave not in escolhidos]
    faltam = tamanho_da_amostra(len(julgados), len(por_processo)) - len(escolhidos)
    escolhidos.update(rng.sample(restantes, faltam))
    return [chave for chave in julgados if chave in escolhidos]


def codificar(candidatos: list[dict[str, str]], semente: int, processo_id: str) -> list[tuple[str, dict[str, str]]]:
    """Candidatos por categoria (embaralhados dentro dela), com codigos neutros (``K01``...)."""
    ordem = list(candidatos)
    random.Random(f"amostra|{semente}|{processo_id}").shuffle(ordem)
    posicao_da_categoria = {categoria: posicao for posicao, categoria in enumerate(CATEGORIES)}
    ordem.sort(key=lambda c: posicao_da_categoria.get(c["categoria"], len(CATEGORIES)))  # estavel: mantem o sorteio
    return [(f"K{posicao:02d}", candidato) for posicao, candidato in enumerate(ordem, start=1)]


def _risco_real(risco: dict[str, str]) -> str:
    """O mesmo que o juiz recebeu: enunciado, causa e consequencia."""
    partes = [risco["risco"]]
    if risco["causa"]:
        partes.append(f"[Causa: {risco['causa']}]")
    if risco["consequencia"]:
        partes.append(f"[Consequência: {risco['consequencia']}]")
    return " ".join(partes)


def gerar_amostra(
    corpus: Path,
    *,
    semente: int = SEMENTE_PADRAO,
    permitir_parcial: bool = False,
    saida: Callable[[str], None] = print,
) -> int:
    """Grava a amostra e a chave; devolve quantos riscos foram sorteados."""
    corpus = corpus.resolve()
    if corpus == ROOT or ROOT in corpus.parents:
        raise AmostraError("A pasta do corpus nao pode ficar dentro do repositorio.")
    saida_dir = corpus / gerar.SAIDA_DIR
    for nome in (AMOSTRA_CSV, CHAVE_CSV):
        if (saida_dir / nome).exists():
            raise AmostraError(f"{nome} ja existe e nao e sobrescrito (pode ter marcacoes). Renomeie-o para sortear de novo.")

    processos = [p.processo_id for p in gerar.ler_processos(corpus)]
    riscos = julgar.ler_riscos_reais(corpus)
    candidatos = julgar.ler_candidatos(saida_dir)
    # do julgamento so se le QUAIS riscos foram julgados; a resposta do juiz nao entra na amostra
    feitos = {
        (linha["processo_id"], linha["risco_id"])
        for linha in conf.ler_tabela(saida_dir / julgar.JULGAMENTO_CSV, ["processo_id", "risco_id"])
    }
    todos = [(pid, risco["risco_id"]) for pid in processos for risco in riscos.get(pid, [])]
    julgados = [chave for chave in todos if chave in feitos]
    if not julgados:
        raise AmostraError(f"Nenhum risco julgado em {julgar.JULGAMENTO_CSV}.")
    if len(julgados) < len(todos) and not permitir_parcial:
        raise AmostraError(
            f"So {len(julgados)} de {len(todos)} riscos reais foram julgados; termine o julgamento antes de "
            "sortear (ou use --permitir-parcial)."
        )

    sorteados = sortear(julgados, semente)
    por_chave = {(pid, risco["risco_id"]): risco for pid in processos for risco in riscos.get(pid, [])}
    codificados = {pid: codificar(candidatos[pid], semente, pid) for pid in dict.fromkeys(pid for pid, _ in sorteados)}
    linhas = [
        {
            "processo_id": pid,
            "risco_id": rid,
            "risco_real": _risco_real(por_chave[(pid, rid)]),
            "candidato_id": codigo,
            "titulo_candidato": candidato["titulo"],
            "causa_candidato": candidato["causa"],
            "consequencia_candidato": candidato["consequencia"],
            "cobre_autor": "",
            "observacao": "",
        }
        for pid, rid in sorteados
        for codigo, candidato in codificados[pid]
    ]
    chave = [
        {"processo_id": pid, "candidato_codigo": codigo, "candidato_id": candidato["candidato_id"], "semente": semente}
        for pid, pares in codificados.items()
        for codigo, candidato in pares
    ]
    conf.gravar_tabela(saida_dir / AMOSTRA_CSV, AMOSTRA_FIELDS, linhas)
    conf.gravar_tabela(saida_dir / CHAVE_CSV, CHAVE_FIELDS, chave)
    saida(
        f"{len(sorteados)} de {len(julgados)} risco(s) julgado(s) sorteado(s), em {len(codificados)} processo(s); "
        f"{len(linhas)} linha(s) em {AMOSTRA_CSV}; semente {semente}."
    )
    saida(f"Chave dos codigos em {CHAVE_CSV} (abrir so depois da conferencia).")
    return len(sorteados)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", required=True, type=Path, help="Pasta do corpus.")
    parser.add_argument("--semente", type=int, default=SEMENTE_PADRAO, help="Semente do sorteio.")
    parser.add_argument("--permitir-parcial", action="store_true", help="Sortear mesmo com julgamento incompleto.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"Pasta do corpus nao encontrada: {corpus}", file=sys.stderr)
        return 2
    try:
        gerar_amostra(corpus, semente=args.semente, permitir_parcial=args.permitir_parcial)
    except (AmostraError, julgar.JulgamentoError, gerar.GeracaoError, conf.ConferenciaError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
