"""Avaliacao retrospectiva: concordancia do painel de juizes com o juiz de referencia.

Le ``julgamento.csv`` (juiz de referencia, o Opus) e os tres arquivos
``painel/julgamento_<juiz>.csv`` e calcula, sem chamar modelo nenhum:

- por juiz: distribuicao de cobertura (catalogo, ia, ambos, nenhum, sem
  julgamento) e numero de candidatos marcados;
- voto da maioria: um par (risco real, candidato) vale como cobertura quando
  pelo menos 2 dos 3 juizes o marcam;
- concordancia por par: kappa de Cohen de cada juiz com a referencia e entre os
  juizes, e kappa de Fleiss dos tres;
- maioria contra a referencia: kappa por par e proporcao de riscos com a mesma
  classe de cobertura e com o mesmo resultado coberto/nao coberto;
- chamadas, falhas, tokens e custo por juiz, com projecao para todos os riscos.

Unidade "par": cada candidato do processo, para cada risco real (marcado ou
nao). O escopo sao os riscos que aparecem em pelo menos um arquivo do painel.
Risco sem julgamento de um juiz fica fora das comparacoes que envolvem esse
juiz e nao conta voto na maioria; com menos de 2 juizes validos, o risco fica
sem maioria e fora da comparacao com a referencia.

Quando o painel alcanca processos fora da analise principal
(``analise_principal`` de ``corpus_casos.csv``), o relatorio sai duas vezes: so
com os da analise principal e com todos os incluidos.

Grava ``painel/painel_votos.csv``: os pares marcados por alguem, com o voto de
cada um. Os exemplos de divergencia evitam os riscos da amostra de conferencia
do autor (que e as cegas), a menos de ``--incluir-amostra``.

Uso:
    python scripts/retro_painel.py --corpus "<pasta do corpus>" [--exemplos N]
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for _caminho in (ROOT, SCRIPTS):
    if str(_caminho) not in sys.path:
        sys.path.insert(0, str(_caminho))

import retro_conferencia as conf  # noqa: E402
import retro_gerar as gerar  # noqa: E402
import retro_juizes as juizes  # noqa: E402
import retro_julgar as julgar  # noqa: E402

REFERENCIA = "opus"
MAIORIA = "maioria"
VOTOS_CSV = "painel_votos.csv"
AMOSTRA_CSV = "amostra_conferencia.csv"
MINIMO_PARA_MAIORIA = 2
SEMENTE_EXEMPLOS = 20261118

Chave = tuple[str, str]
Votos = dict[Chave, "set[str] | None"]  # None = sem julgamento


class PainelError(Exception):
    """Falha que impede calcular as medidas do painel."""


# --------------------------------------------------------------------------
# Medidas
# --------------------------------------------------------------------------


def kappa_cohen(a: list[bool], b: list[bool]) -> float | None:
    """Kappa de Cohen de duas classificacoes binarias; ``None`` se nao ha itens ou nao ha variacao."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    observado = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    esperado = pa * pb + (1 - pa) * (1 - pb)
    if esperado == 1:
        return None
    return (observado - esperado) / (1 - esperado)


def kappa_fleiss(itens: list[tuple[bool, ...]]) -> float | None:
    """Kappa de Fleiss para o mesmo numero de avaliadores por item, duas categorias."""
    if not itens:
        return None
    m = len(itens[0])
    if m < 2 or any(len(item) != m for item in itens):
        return None
    sins = [sum(item) for item in itens]
    acordo = sum((k * (k - 1) + (m - k) * (m - k - 1)) / (m * (m - 1)) for k in sins) / len(itens)
    p_sim = sum(sins) / (len(itens) * m)
    esperado = p_sim**2 + (1 - p_sim) ** 2
    if esperado == 1:
        return None
    return (acordo - esperado) / (1 - esperado)


def concordancia(a: list[bool], b: list[bool]) -> float | None:
    return sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else None


def voto_da_maioria(votos: list[set[str] | None]) -> set[str] | None:
    """Candidatos marcados por pelo menos 2 juizes; ``None`` com menos de 2 juizes validos."""
    validos = [voto for voto in votos if voto is not None]
    if len(validos) < MINIMO_PARA_MAIORIA:
        return None
    contagem = Counter(candidato for voto in validos for candidato in voto)
    return {candidato for candidato, quantos in contagem.items() if quantos >= MINIMO_PARA_MAIORIA}


# --------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------


def ler_votos(caminho: Path) -> Votos:
    """``{(processo, risco): candidatos marcados}``; ``None`` onde o juiz ficou sem julgamento."""
    votos: Votos = {}
    for linha in conf.ler_tabela(caminho, ["processo_id", "risco_id", "candidatos_cobertos", "cobertura"]):
        chave = (linha["processo_id"], linha["risco_id"])
        if chave in votos:
            raise PainelError(f"{caminho.name}: risco repetido: {chave[0]} / {chave[1]}")
        if linha["cobertura"] == julgar.SEM_JULGAMENTO:
            votos[chave] = None
        else:
            partes = linha["candidatos_cobertos"].split(julgar.SEPARADOR.strip())
            votos[chave] = {parte.strip() for parte in partes if parte.strip()}
    return votos


def processos_da_analise_principal(corpus: Path) -> set[str]:
    """Processos incluidos com ``analise_principal=sim`` em ``corpus_casos.csv``."""
    linhas = conf.ler_tabela(corpus / gerar.CASOS_CSV, ["processo_id", "incluir", "analise_principal"])
    return {
        linha["processo_id"]
        for linha in linhas
        if conf._sim_nao(linha["incluir"]) == "sim" and conf._sim_nao(linha["analise_principal"]) == "sim"
    }


def ler_custos(saida_dir: Path, nome: str, processos: set[str] | None = None) -> dict:
    """Chamadas, falhas, tokens e custo (US$) de um juiz, de ``painel/chamadas_<juiz>.csv``."""
    caminho = julgar.chamadas_do_painel(saida_dir, nome)
    linhas = conf.ler_tabela(caminho, julgar.CHAMADAS_FIELDS) if caminho.is_file() else []
    if processos is not None:
        linhas = [linha for linha in linhas if linha["processo_id"] in processos]
    juiz = juizes.JUIZES[nome]
    entrada = sum(int(linha["tokens_entrada"] or 0) for linha in linhas)
    saida = sum(int(linha["tokens_saida"] or 0) for linha in linhas)
    return {
        "chamadas": len(linhas),
        "riscos": len({(linha["processo_id"], linha["risco_id"]) for linha in linhas}),
        "situacoes": Counter(linha["situacao"] for linha in linhas),
        "tokens_entrada": entrada,
        "tokens_saida": saida,
        "custo": (entrada * juiz.preco_entrada + saida * juiz.preco_saida) / 1_000_000,
    }


# --------------------------------------------------------------------------
# Analise
# --------------------------------------------------------------------------


def _classe(marcados: set[str] | None, origem: dict[str, str]) -> str:
    if marcados is None:
        return julgar.SEM_JULGAMENTO
    return julgar.derivar_cobertura([origem[candidato] for candidato in marcados])


def _pares(a: Votos, b: Votos, escopo: list[Chave], ids: dict[str, list[str]]) -> tuple[list[bool], list[bool]]:
    """Votos dos dois, par a par, nos riscos do escopo em que ambos julgaram."""
    de_a, de_b = [], []
    for chave in escopo:
        if a.get(chave) is None or b.get(chave) is None:
            continue
        for candidato in ids[chave[0]]:
            de_a.append(candidato in a[chave])
            de_b.append(candidato in b[chave])
    return de_a, de_b


def analisar(corpus: Path, nomes: list[str] | None = None, processos: set[str] | None = None) -> dict:
    """Todas as medidas do painel; nao grava nada. ``processos`` restringe a analise a esses processos."""
    corpus = corpus.resolve()
    saida_dir = corpus / gerar.SAIDA_DIR
    nomes = list(nomes or juizes.JUIZES)
    candidatos = julgar.ler_candidatos(saida_dir)
    ids = {pid: [c["candidato_id"] for c in lista] for pid, lista in candidatos.items()}
    origem = {c["candidato_id"]: c["origem"] for lista in candidatos.values() for c in lista}
    referencia = ler_votos(saida_dir / julgar.JULGAMENTO_CSV)
    faltando = [nome for nome in nomes if not julgar.arquivo_do_painel(saida_dir, nome).is_file()]
    if faltando:
        raise PainelError(f"Falta(m) o(s) arquivo(s) do painel de: {', '.join(faltando)}.")
    painel = {nome: ler_votos(julgar.arquivo_do_painel(saida_dir, nome)) for nome in nomes}
    for nome, votos in {REFERENCIA: referencia, **painel}.items():
        for (pid, rid), marcados in votos.items():
            if marcados and not marcados <= set(ids.get(pid, [])):
                raise PainelError(f"{nome}: {pid} / {rid} cita candidato que nao esta em {gerar.CANDIDATOS_CSV}.")
    fora = sorted({chave for votos in painel.values() for chave in votos} - set(referencia))
    if fora:
        raise PainelError(f"Risco do painel sem julgamento de referencia: {fora[0][0]} / {fora[0][1]}")
    if processos is not None:
        referencia = {chave: voto for chave, voto in referencia.items() if chave[0] in processos}
        painel = {n: {chave: voto for chave, voto in votos.items() if chave[0] in processos} for n, votos in painel.items()}
    escopo = [chave for chave in referencia if any(chave in votos for votos in painel.values())]
    if not escopo:
        raise PainelError("Nenhum risco julgado pelo painel.")
    # risco do escopo que um juiz ainda nao tentou tambem esta sem julgamento
    painel = {nome: {chave: votos.get(chave) for chave in escopo} for nome, votos in painel.items()}
    maioria: Votos = {chave: voto_da_maioria([painel[nome][chave] for nome in nomes]) for chave in escopo}
    todos = {REFERENCIA: referencia, **painel, MAIORIA: maioria}

    por_juiz = {}
    for nome, votos in todos.items():
        validos = [votos[chave] for chave in escopo if votos[chave] is not None]
        por_juiz[nome] = {
            "classes": Counter(_classe(votos[chave], origem) for chave in escopo),
            "marcados": sum(len(marcados) for marcados in validos),
            "riscos_julgados": len(validos),
        }

    def comparar(a: str, b: str) -> dict:
        de_a, de_b = _pares(todos[a], todos[b], escopo, ids)
        return {"pares": len(de_a), "concordancia": concordancia(de_a, de_b), "kappa": kappa_cohen(de_a, de_b)}

    comparacoes = {(nome, REFERENCIA): comparar(nome, REFERENCIA) for nome in nomes}
    comparacoes.update({(a, b): comparar(a, b) for i, a in enumerate(nomes) for b in nomes[i + 1 :]})
    comparacoes[(MAIORIA, REFERENCIA)] = comparar(MAIORIA, REFERENCIA)

    completos = [chave for chave in escopo if all(painel[nome][chave] is not None for nome in nomes)]
    itens = [tuple(c in painel[nome][chave] for nome in nomes) for chave in completos for c in ids[chave[0]]]
    com_maioria = [chave for chave in escopo if maioria[chave] is not None]
    classes = [(_classe(maioria[chave], origem), _classe(referencia[chave], origem)) for chave in com_maioria]
    divergencias = [
        {
            "processo_id": chave[0],
            "risco_id": chave[1],
            "candidato_id": candidato,
            REFERENCIA: candidato in referencia[chave],
            **{nome: (None if painel[nome][chave] is None else candidato in painel[nome][chave]) for nome in nomes},
            MAIORIA: candidato in maioria[chave],
        }
        for chave in com_maioria
        for candidato in ids[chave[0]]
        if (candidato in maioria[chave]) != (candidato in referencia[chave])
    ]
    return {
        "nomes": nomes,
        "escopo": escopo,
        "total_de_riscos": len(referencia),
        "candidatos": candidatos,
        "votos": todos,
        "por_juiz": por_juiz,
        "comparacoes": comparacoes,
        "fleiss": {"pares": len(itens), "riscos": len(completos), "kappa": kappa_fleiss(itens)},
        "maioria_x_referencia": {
            "riscos": len(com_maioria),
            "sem_maioria": len(escopo) - len(com_maioria),
            "mesma_classe": sum(1 for a, b in classes if a == b),
            "mesmo_coberto": sum(1 for a, b in classes if (a == "nenhum") == (b == "nenhum")),
        },
        "divergencias": divergencias,
        "processos": len({pid for pid, _ in escopo}),
        "custos": {nome: ler_custos(saida_dir, nome, processos) for nome in nomes},
    }


def escolher_exemplos(divergencias: list[dict], quantos: int, evitar: set[Chave]) -> list[dict]:
    """Ate ``quantos`` pares divergentes, sorteados com semente fixa e, se der, de riscos diferentes."""
    ordem = [d for d in divergencias if (d["processo_id"], d["risco_id"]) not in evitar]
    random.Random(SEMENTE_EXEMPLOS).shuffle(ordem)
    vistos: set[Chave] = set()
    primeiros, repetidos = [], []
    for d in ordem:
        chave = (d["processo_id"], d["risco_id"])
        (repetidos if chave in vistos else primeiros).append(d)
        vistos.add(chave)
    return (primeiros + repetidos)[:quantos]


def linhas_de_votos(resultado: dict) -> list[dict[str, str]]:
    """Pares marcados por alguem (referencia ou juiz do painel), com o voto de cada um."""
    nomes, votos = resultado["nomes"], resultado["votos"]
    colunas = [REFERENCIA, *nomes, MAIORIA]

    def texto(valor: bool | None) -> str:
        return julgar.SEM_JULGAMENTO if valor is None else ("sim" if valor else "nao")

    linhas = []
    for pid, rid in resultado["escopo"]:
        for candidato in resultado["candidatos"][pid]:
            cid = candidato["candidato_id"]
            marcas = {n: (None if votos[n][(pid, rid)] is None else cid in votos[n][(pid, rid)]) for n in colunas}
            if any(marcas.values()):
                linhas.append({"processo_id": pid, "risco_id": rid, "candidato_id": cid, **{n: texto(v) for n, v in marcas.items()}})
    return linhas


def gravar_votos(saida_dir: Path, resultado: dict) -> Path:
    """Grava ``painel/painel_votos.csv`` com as linhas de ``linhas_de_votos``."""
    caminho = saida_dir / julgar.PAINEL_DIR / VOTOS_CSV
    colunas = ["processo_id", "risco_id", "candidato_id", REFERENCIA, *resultado["nomes"], MAIORIA]
    conf.gravar_tabela(caminho, colunas, linhas_de_votos(resultado))
    return caminho


# --------------------------------------------------------------------------
# Relatorio
# --------------------------------------------------------------------------


def _n(valor: float | None, formato: str = ".3f") -> str:
    return "n/d" if valor is None else format(valor, formato).replace(".", ",")


def _pct(parte: int, total: int) -> str:
    return f"{parte}/{total} ({_n(100 * parte / total, '.1f')}%)" if total else "0/0"


def _com_pct(parte: int, total: int) -> str:
    return f"{parte} ({_n(100 * parte / total, '.1f')}%)"


def relatorio(resultado: dict) -> list[str]:
    nomes = resultado["nomes"]
    modelo = {nome: juizes.JUIZES[nome].modelo for nome in nomes}
    escopo, total = len(resultado["escopo"]), resultado["total_de_riscos"]
    linhas = [
        f"Escopo: {escopo} risco(s) reais julgados pelo painel, em {resultado['processos']} processo(s), "
        f"de {total} com julgamento de referencia.",
        "",
    ]

    linhas.append("== Por juiz: cobertura e candidatos marcados ==")
    colunas = ["catalogo", "so ia", "ambos", "coberto", "nenhum", "sem julg."]
    linhas.append(f"{'juiz':<34}" + "".join(f"{coluna:>15}" for coluna in colunas) + f"{'marcados':>10}{'por risco':>11}")
    for nome in (REFERENCIA, *nomes, MAIORIA):
        dados = resultado["por_juiz"][nome]
        classes = dados["classes"]
        rotulo = f"{nome} ({modelo[nome]})" if nome in modelo else nome
        media = dados["marcados"] / dados["riscos_julgados"] if dados["riscos_julgados"] else None
        coberto = classes["catalogo"] + classes["ia"] + classes["ambos"]
        valores = [classes["catalogo"], classes["ia"], classes["ambos"], coberto, classes["nenhum"]]
        linhas.append(
            f"{rotulo:<34}" + "".join(f"{_com_pct(valor, escopo):>15}" for valor in valores)
            + f"{classes[julgar.SEM_JULGAMENTO]:>15}{dados['marcados']:>10}{_n(media, '.2f'):>11}"
        )
    linhas.append("(coberto = catalogo + so ia + ambos; % sobre os riscos do escopo; na maioria, 'sem julg.' = menos de 2 juizes validos)")

    linhas += ["", "== Concordancia por par (risco real, candidato) =="]
    linhas.append(f"{'comparacao':<28}{'pares':>8}{'concordancia':>14}{'kappa':>9}")
    for (a, b), dados in resultado["comparacoes"].items():
        linhas.append(f"{a + ' x ' + b:<28}{dados['pares']:>8}{_n(dados['concordancia']):>14}{_n(dados['kappa']):>9}")
    fleiss = resultado["fleiss"]
    linhas.append(f"Kappa de Fleiss dos {len(nomes)} juizes: {_n(fleiss['kappa'])} ({fleiss['pares']} pares, {fleiss['riscos']} riscos com todos os votos)")

    mxr = resultado["maioria_x_referencia"]
    geral = resultado["comparacoes"][(MAIORIA, REFERENCIA)]
    linhas += ["", "== Maioria do painel contra a referencia =="]
    linhas.append(f"Kappa por par: {_n(geral['kappa'])} (concordancia {_n(geral['concordancia'])}, {geral['pares']} pares)")
    linhas.append(f"Mesma classe de cobertura: {_pct(mxr['mesma_classe'], mxr['riscos'])} dos riscos")
    linhas.append(f"Mesmo resultado coberto/nao coberto: {_pct(mxr['mesmo_coberto'], mxr['riscos'])} dos riscos")
    linhas.append(f"Riscos sem maioria (fora desta comparacao): {mxr['sem_maioria']}")
    linhas.append(f"Pares em que a maioria e a referencia divergem: {len(resultado['divergencias'])}")

    linhas += ["", "== Falhas, tokens e custo por juiz =="]
    linhas.append(
        f"{'juiz':<34}{'sem julg.':>10}{'chamadas':>10}{'invalidas':>11}{'erros':>7}{'tok. entrada':>14}{'tok. saida':>12}"
        f"{'custo US$':>11}{f'proj. {total} US$':>16}"
    )
    soma = projecao_total = 0.0
    for nome in nomes:
        custo = resultado["custos"][nome]
        projecao = custo["custo"] * total / custo["riscos"] if custo["riscos"] else None
        soma += custo["custo"]
        projecao_total += projecao or 0.0
        linhas.append(
            f"{nome + ' (' + modelo[nome] + ')':<34}{resultado['por_juiz'][nome]['classes'][julgar.SEM_JULGAMENTO]:>10}"
            f"{custo['chamadas']:>10}{custo['situacoes']['resposta invalida']:>11}{custo['situacoes']['erro do provedor']:>7}"
            f"{custo['tokens_entrada']:>14}{custo['tokens_saida']:>12}{_n(custo['custo'], '.4f'):>11}{_n(projecao, '.2f'):>16}"
        )
    linhas.append(f"{'total':<34}{'':>64}{_n(soma, '.4f'):>11}{_n(projecao_total, '.2f'):>16}")
    linhas.append("(custo = tokens x preco de tabela na data da selecao; a projecao e proporcional ao numero de riscos)")
    return linhas


def exemplos(resultado: dict, corpus: Path, quantos: int, evitar: set[Chave]) -> list[str]:
    riscos = {(pid, r["risco_id"]): r for pid, lista in julgar.ler_riscos_reais(corpus).items() for r in lista}
    candidato = {c["candidato_id"]: c for lista in resultado["candidatos"].values() for c in lista}
    escolhidos = escolher_exemplos(resultado["divergencias"], quantos, evitar)
    linhas = ["", f"== {len(escolhidos)} exemplo(s) de par em que a maioria e a referencia divergem =="]

    def voto(valor: bool | None) -> str:
        return julgar.SEM_JULGAMENTO if valor is None else ("cobre" if valor else "nao cobre")

    for numero, d in enumerate(escolhidos, start=1):
        risco, c = riscos[(d["processo_id"], d["risco_id"])], candidato[d["candidato_id"]]
        linhas += [
            "",
            f"[{numero}] {d['processo_id']} / {d['risco_id']}",
            f"  Risco real: {risco['risco']}",
            f"    Causa: {risco['causa'] or julgar.NAO_INFORMADA} | Consequencia: {risco['consequencia'] or julgar.NAO_INFORMADA}",
            f"  Candidato {d['candidato_id']} ({c['origem']}): {c['titulo']}",
            f"    Causa: {c['causa'] or julgar.NAO_INFORMADA} | Consequencia: {c['consequencia'] or julgar.NAO_INFORMADA}",
            "  Votos: " + "; ".join(f"{nome}: {voto(d[nome])}" for nome in (REFERENCIA, *resultado["nomes"], MAIORIA)),
        ]
    return linhas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", required=True, type=Path, help="Pasta do corpus.")
    parser.add_argument("--exemplos", type=int, default=5, help="Quantos pares divergentes mostrar (0 = nenhum).")
    parser.add_argument(
        "--incluir-amostra",
        action="store_true",
        help="Permitir exemplos de riscos da amostra de conferencia do autor (revela o voto da referencia).",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"Pasta do corpus nao encontrada: {corpus}", file=sys.stderr)
        return 2
    try:
        resultado = analisar(corpus)
        saida_dir = corpus / gerar.SAIDA_DIR
        principal = processos_da_analise_principal(corpus)
        no_escopo = {pid for pid, _ in resultado["escopo"]}
        if principal & no_escopo and not no_escopo <= principal:
            print("######## ANALISE PRINCIPAL (processos com analise_principal=sim) ########")
            print("\n".join(relatorio(analisar(corpus, processos=principal))))
            print("\n######## TODOS OS PROCESSOS INCLUIDOS ########")
        print("\n".join(relatorio(resultado)))
        if args.exemplos > 0:
            amostra = saida_dir / AMOSTRA_CSV
            evitar: set[Chave] = set()
            if amostra.is_file() and not args.incluir_amostra:
                evitar = {(l["processo_id"], l["risco_id"]) for l in conf.ler_tabela(amostra, ["processo_id", "risco_id"])}
            print("\n".join(exemplos(resultado, corpus, args.exemplos, evitar)))
            if evitar:
                print("\n(os exemplos evitam os riscos da amostra de conferencia do autor)")
        print(f"\nVotos por par em {gravar_votos(saida_dir, resultado)}")
    except (PainelError, julgar.JulgamentoError, conf.ConferenciaError, gerar.GeracaoError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
