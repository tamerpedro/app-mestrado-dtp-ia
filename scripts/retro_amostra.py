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

Modo ``--reduzida``: mesma conferencia, com menos linhas. Usa os MESMOS riscos
ja sorteados (lidos de ``amostra_conferencia.csv``) e os mesmos codigos neutros
(lidos de ``amostra_chave.csv``) e grava ``amostra_conferencia_reduzida.csv`` e
``amostra_chave_reduzida.csv``. Cinco riscos, sorteados com a semente e de pelo
menos quatro processos, mantem a lista completa de candidatos e ficam no fim do
arquivo. Nos demais a lista tem 12 candidatos, sem repeticao:

- todos os que o juiz marcou para aquele risco em ``julgamento.csv``;
- os mais parecidos com o risco real, ate completar 9: similaridade de cosseno
  TF-IDF entre o texto do risco real (risco, causa, consequencia) e o do
  candidato (titulo, causa, consequencia), calculada aqui, sem rede, sobre os
  candidatos do mesmo processo;
- 3 sorteados entre os restantes, com a semente.

Se o juiz marcou mais de 9, entram todos os marcados e os 3 sorteados. Dentro
de cada risco os candidatos vem embaralhados, para a posicao nao indicar por
que cada um entrou; o motivo (``juiz``, ``similaridade``, ``sorteio`` ou
``lista completa``) e a posicao no ranking de similaridade ficam so na chave.

Uso:
    python scripts/retro_amostra.py --corpus "<pasta do corpus>"
    python scripts/retro_amostra.py --corpus "<pasta do corpus>" --reduzida
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import unicodedata
from collections import Counter
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


# --------------------------------------------------------------------------
# Amostra reduzida
# --------------------------------------------------------------------------

AMOSTRA_REDUZIDA_CSV = "amostra_conferencia_reduzida.csv"
CHAVE_REDUZIDA_CSV = "amostra_chave_reduzida.csv"
CHAVE_REDUZIDA_FIELDS = [
    "processo_id",
    "risco_id",
    "candidato_codigo",
    "candidato_id",
    "entrada",  # juiz | similaridade | sorteio | lista completa
    "posicao_similaridade",  # 1 = o candidato do processo mais parecido com o risco real
    "similaridade",
    "semente",
]
ENTRADA_JUIZ = "juiz"
ENTRADA_SIMILARIDADE = "similaridade"
ENTRADA_SORTEIO = "sorteio"
ENTRADA_COMPLETA = "lista completa"
ATE_SIMILARIDADE = 9  # marcados pelo juiz + mais parecidos
SORTEADOS_POR_RISCO = 3
LISTAS_COMPLETAS = 5
PROCESSOS_MIN_COMPLETAS = 4

_PALAVRA_RE = re.compile(r"[a-z0-9]{2,}")
# palavras gramaticais do portugues, ja sem acento; nao entram no calculo da similaridade
_VAZIAS = frozenset(
    "ao aos as com como da das de do dos em entre na nas no nos ou para pela pelas pelo pelos por que "
    "se sem sob sobre um uma umas uns nao mais ser seu sua seus suas foi sao ha ja este esta esse essa "
    "isso isto aquele aquela quando onde qual quais tambem".split()
)


def _palavras(texto: str) -> list[str]:
    """Palavras em minusculas e sem acento, sem as gramaticais."""
    sem_acento = "".join(c for c in unicodedata.normalize("NFKD", texto.casefold()) if not unicodedata.combining(c))
    return [palavra for palavra in _PALAVRA_RE.findall(sem_acento) if palavra not in _VAZIAS]


def similaridades(consulta: str, documentos: list[str]) -> list[float]:
    """Cosseno TF-IDF entre ``consulta`` e cada documento.

    Frequencia bruta do termo vezes ``ln((1 + n) / (1 + df)) + 1``, com ``n`` e
    ``df`` contados sobre os documentos mais a consulta.
    """
    contagens = [Counter(_palavras(texto)) for texto in [consulta, *documentos]]
    df = Counter(termo for contagem in contagens for termo in contagem)
    idf = {termo: math.log((1 + len(contagens)) / (1 + quantos)) + 1 for termo, quantos in df.items()}
    vetores = [{termo: vezes * idf[termo] for termo, vezes in contagem.items()} for contagem in contagens]
    normas = [math.sqrt(sum(peso * peso for peso in vetor.values())) for vetor in vetores]
    da_consulta, norma_da_consulta = vetores[0], normas[0]
    return [
        sum(peso * da_consulta.get(termo, 0.0) for termo, peso in vetor.items()) / (norma * norma_da_consulta)
        if norma and norma_da_consulta
        else 0.0
        for vetor, norma in zip(vetores[1:], normas[1:])
    ]


def ranking_de_similaridade(risco: dict[str, str], candidatos: list[dict[str, str]]) -> dict[str, tuple[int, float]]:
    """``{candidato_id: (posicao, similaridade)}``; empates ficam na ordem de ``candidatos.csv``."""
    consulta = " ".join(risco[campo] for campo in ("risco", "causa", "consequencia"))
    documentos = [" ".join(c[campo] for campo in ("titulo", "causa", "consequencia")) for c in candidatos]
    valores = similaridades(consulta, documentos)
    ordem = sorted(range(len(candidatos)), key=lambda i: (-valores[i], i))
    return {candidatos[i]["candidato_id"]: (posicao, valores[i]) for posicao, i in enumerate(ordem, start=1)}


def compor_lista(
    candidatos: list[dict[str, str]],
    marcados: set[str],
    ranking: dict[str, tuple[int, float]],
    rng: random.Random,
) -> list[tuple[dict[str, str], str]]:
    """``(candidato, entrada)`` da lista reduzida de um risco, sem repeticao e ainda sem embaralhar."""
    do_juiz = [c for c in candidatos if c["candidato_id"] in marcados]
    restantes = sorted(
        (c for c in candidatos if c["candidato_id"] not in marcados), key=lambda c: ranking[c["candidato_id"]][0]
    )
    faltam = max(0, ATE_SIMILARIDADE - len(do_juiz))
    parecidos, restantes = restantes[:faltam], restantes[faltam:]
    sorteados = rng.sample(restantes, min(SORTEADOS_POR_RISCO, len(restantes)))
    return (
        [(c, ENTRADA_JUIZ) for c in do_juiz]
        + [(c, ENTRADA_SIMILARIDADE) for c in parecidos]
        + [(c, ENTRADA_SORTEIO) for c in sorteados]
    )


def sortear_listas_completas(sorteados: list[tuple[str, str]], semente: int) -> set[tuple[str, str]]:
    """Riscos que mantem a lista completa: 5, de pelo menos 4 processos (ou de quantos houver)."""
    quantos = min(LISTAS_COMPLETAS, len(sorteados))
    processos_min = min(PROCESSOS_MIN_COMPLETAS, quantos, len({pid for pid, _ in sorteados}))
    rng = random.Random(f"reduzida|completas|{semente}")
    while True:
        escolhidos = rng.sample(sorteados, quantos)
        if len({pid for pid, _ in escolhidos}) >= processos_min:
            return set(escolhidos)


def gerar_amostra_reduzida(corpus: Path, *, saida: Callable[[str], None] = print) -> int:
    """Grava a amostra reduzida e a chave dela; devolve quantos riscos ela tem."""
    corpus = corpus.resolve()
    if corpus == ROOT or ROOT in corpus.parents:
        raise AmostraError("A pasta do corpus nao pode ficar dentro do repositorio.")
    saida_dir = corpus / gerar.SAIDA_DIR
    for nome in (AMOSTRA_REDUZIDA_CSV, CHAVE_REDUZIDA_CSV):
        if (saida_dir / nome).exists():
            raise AmostraError(f"{nome} ja existe e nao e sobrescrito (pode ter marcacoes). Renomeie-o para gerar de novo.")
    for nome in (AMOSTRA_CSV, CHAVE_CSV):
        if not (saida_dir / nome).is_file():
            raise AmostraError(f"{nome} nao encontrado: a amostra reduzida usa os riscos ja sorteados.")

    # a chave atual nao traz o risco_id: os riscos sorteados saem da amostra atual; os codigos e a semente, da chave
    sorteados = list(
        dict.fromkeys(
            (linha["processo_id"], linha["risco_id"])
            for linha in conf.ler_tabela(saida_dir / AMOSTRA_CSV, ["processo_id", "risco_id"])
        )
    )
    chave_atual = conf.ler_tabela(saida_dir / CHAVE_CSV, CHAVE_FIELDS)
    sementes = {linha["semente"] for linha in chave_atual}
    if len(sementes) != 1 or not next(iter(sementes)).isdigit():
        raise AmostraError(f"{CHAVE_CSV}: semente ausente ou com mais de um valor.")
    semente = int(next(iter(sementes)))
    codigo_de = {(linha["processo_id"], linha["candidato_id"]): linha["candidato_codigo"] for linha in chave_atual}

    riscos = {
        (pid, risco["risco_id"]): risco for pid, lista in julgar.ler_riscos_reais(corpus).items() for risco in lista
    }
    candidatos = julgar.ler_candidatos(saida_dir)
    marcados_de = {
        (linha["processo_id"], linha["risco_id"]): {
            parte.strip() for parte in linha["candidatos_cobertos"].split(julgar.SEPARADOR.strip()) if parte.strip()
        }
        for linha in conf.ler_tabela(saida_dir / julgar.JULGAMENTO_CSV, ["processo_id", "risco_id", "candidatos_cobertos"])
    }
    for pid, rid in sorteados:
        if (pid, rid) not in riscos or (pid, rid) not in marcados_de or pid not in candidatos:
            raise AmostraError(f"{pid} / {rid}: risco da amostra sem enunciado, sem julgamento ou sem candidatos.")
        ids = {c["candidato_id"] for c in candidatos[pid]}
        if not marcados_de[(pid, rid)] <= ids:
            raise AmostraError(f"{pid} / {rid}: {julgar.JULGAMENTO_CSV} cita candidato que nao esta em {gerar.CANDIDATOS_CSV}.")
        if any((pid, cid) not in codigo_de for cid in ids):
            raise AmostraError(f"{pid}: candidato sem codigo em {CHAVE_CSV}.")

    completas = sortear_listas_completas(sorteados, semente)
    ordem = [chave for chave in sorteados if chave not in completas] + [chave for chave in sorteados if chave in completas]
    linhas, chave = [], []
    for pid, rid in ordem:
        ranking = ranking_de_similaridade(riscos[(pid, rid)], candidatos[pid])
        rng = random.Random(f"reduzida|{semente}|{pid}|{rid}")
        if (pid, rid) in completas:
            lista = [(c, ENTRADA_COMPLETA) for c in candidatos[pid]]
        else:
            lista = compor_lista(candidatos[pid], marcados_de[(pid, rid)], ranking, rng)
        rng.shuffle(lista)
        for candidato, entrada in lista:
            codigo = codigo_de[(pid, candidato["candidato_id"])]
            posicao, valor = ranking[candidato["candidato_id"]]
            linhas.append(
                {
                    "processo_id": pid,
                    "risco_id": rid,
                    "risco_real": _risco_real(riscos[(pid, rid)]),
                    "candidato_id": codigo,
                    "titulo_candidato": candidato["titulo"],
                    "causa_candidato": candidato["causa"],
                    "consequencia_candidato": candidato["consequencia"],
                    "cobre_autor": "",
                    "observacao": "",
                }
            )
            chave.append(
                {
                    "processo_id": pid,
                    "risco_id": rid,
                    "candidato_codigo": codigo,
                    "candidato_id": candidato["candidato_id"],
                    "entrada": entrada,
                    "posicao_similaridade": posicao,
                    "similaridade": f"{valor:.4f}",
                    "semente": semente,
                }
            )
    conf.gravar_tabela(saida_dir / AMOSTRA_REDUZIDA_CSV, AMOSTRA_FIELDS, linhas)
    conf.gravar_tabela(saida_dir / CHAVE_REDUZIDA_CSV, CHAVE_REDUZIDA_FIELDS, chave)
    saida(
        f"{len(ordem)} risco(s) da amostra atual; {len(completas)} com a lista completa (no fim do arquivo); "
        f"{len(linhas)} linha(s) em {AMOSTRA_REDUZIDA_CSV}; semente {semente}."
    )
    saida(f"Chave em {CHAVE_REDUZIDA_CSV} (abrir so depois da conferencia).")
    return len(ordem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", required=True, type=Path, help="Pasta do corpus.")
    parser.add_argument("--semente", type=int, default=SEMENTE_PADRAO, help="Semente do sorteio.")
    parser.add_argument("--permitir-parcial", action="store_true", help="Sortear mesmo com julgamento incompleto.")
    parser.add_argument(
        "--reduzida",
        action="store_true",
        help="Gerar a amostra reduzida a partir da amostra ja sorteada (usa a semente gravada na chave).",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"Pasta do corpus nao encontrada: {corpus}", file=sys.stderr)
        return 2
    try:
        if args.reduzida:
            gerar_amostra_reduzida(corpus)
        else:
            gerar_amostra(corpus, semente=args.semente, permitir_parcial=args.permitir_parcial)
    except (AmostraError, julgar.JulgamentoError, gerar.GeracaoError, conf.ConferenciaError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
