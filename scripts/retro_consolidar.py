"""Avaliacao retrospectiva: consolidacao dos resultados (Parte 3).

Nao chama modelo nenhum. Le o corpus, os candidatos, o julgamento principal (o
Opus, ``julgamento.csv``), a classificacao dos riscos de IA (``riscos_ia.csv``),
o log de geracao e o painel de juizes, e grava em
``<corpus>/retrospectiva/resultados/`` dois conjuntos de tabelas:

- ``principal/``: processos com ``incluir=sim`` e ``analise_principal=sim``;
- ``sensibilidade/``: todos os processos com ``incluir=sim``.

Em todas as tabelas a coluna ``grupo`` e o grupo do processo em
``corpus_casos.csv`` (``externo``, ``dataprev``); os grupos nunca sao somados.
O agrupamento dos riscos de IA parecidos (coluna ``grupo`` de ``riscos_ia.csv``)
aparece como ``grupo_ia``.

Tabelas do julgamento principal: ``por_processo``, ``por_tipo``,
``por_categoria``, ``nao_cobertos``, ``ia_sem_correspondente`` e
``ia_cobertura_incremental``. Tabelas de robustez (maioria do painel, 2 de 3):
``robustez_cobertura``, ``robustez_concordancia``, ``robustez_por_processo``,
``robustez_ia_sem_correspondente`` e ``divergencias_maioria_opus``.

Definicoes:

- classe de cobertura de um risco real: ``catalogo`` (so candidatos do catalogo
  marcados), ``ia`` (so riscos de IA), ``ambos`` ou ``nenhum``;
- cobertura do catalogo = so catalogo + ambos; cobertura incremental da IA = so
  IA; cobertura total = catalogo + so IA + ambos;
- risco de IA sem correspondente: nenhum risco real do processo o tem entre os
  candidatos marcados; grupo de IA sem correspondente: nenhum risco do grupo tem
  correspondente.

As medidas do painel vem de ``retro_painel.analisar``; os numeros sao os mesmos
que ``retro_painel.py`` imprime.

Uso:
    python scripts/retro_consolidar.py --corpus "<pasta do corpus>"
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
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
import retro_painel as painel  # noqa: E402
from src.catalog_mix import ORIGIN_AI, ORIGIN_CATALOG  # noqa: E402
from src.domain import CATEGORIES  # noqa: E402

RESULTADOS_DIR = "resultados"
CONJUNTOS = ("principal", "sensibilidade")
SEM = julgar.SEM_JULGAMENTO
CLASSES = ("catalogo", "ia", "ambos", "nenhum")
SOMAVEIS = (
    "riscos_reais", "so_catalogo", "so_ia", "ambos", "coberto", "nenhum", "sem_julgamento",
    "ia_gerados", "ia_grupos", "ia_nomeia_objeto", "ia_especifico_do_objeto",
    "ia_sem_correspondente", "ia_grupos_sem_correspondente",
)  # fmt: skip

Chave = tuple[str, str]


class ConsolidacaoError(Exception):
    """Falha que impede consolidar os resultados."""


@dataclass
class Dados:
    casos: dict[str, dict[str, str]]  # processo -> grupo, tipo, principal (ordem do arquivo)
    riscos: dict[str, list[dict[str, str]]]
    candidatos: dict[str, list[dict[str, str]]]
    ia: dict[str, dict[str, str]]  # candidato_id -> linha de riscos_ia.csv
    log: dict[str, dict[str, str]]  # processo -> linha mais recente com situacao "gerado"
    origem: dict[str, str]


# --------------------------------------------------------------------------
# Leitura e validacao
# --------------------------------------------------------------------------


def carregar(corpus: Path) -> Dados:
    """Le as entradas (somente leitura) e para se faltar coluna ou se os arquivos nao fecharem entre si."""
    saida_dir = corpus / gerar.SAIDA_DIR
    linhas = conf.ler_tabela(
        corpus / gerar.CASOS_CSV, ["processo_id", "grupo", "tipo_sugerido", "incluir", "analise_principal"]
    )
    casos: dict[str, dict[str, str]] = {}
    for linha in linhas:
        if conf._sim_nao(linha["incluir"]) != "sim":
            continue
        pid, grupo, tipo = linha["processo_id"], conf._sim_nao(linha["grupo"]), conf._sim_nao(linha["tipo_sugerido"])
        if pid in casos:
            raise ConsolidacaoError(f"{gerar.CASOS_CSV}: processo repetido: {pid}")
        if not grupo:
            raise ConsolidacaoError(f"{gerar.CASOS_CSV}: {pid} sem grupo.")
        if tipo not in gerar.TIPOS:
            raise ConsolidacaoError(f"{gerar.CASOS_CSV}: {pid} com tipo_sugerido desconhecido '{linha['tipo_sugerido']}'.")
        casos[pid] = {"grupo": grupo, "tipo": tipo, "principal": conf._sim_nao(linha["analise_principal"])}
    if not casos:
        raise ConsolidacaoError(f"Nenhum processo com incluir=sim em {gerar.CASOS_CSV}.")

    riscos = julgar.ler_riscos_reais(corpus)
    candidatos = julgar.ler_candidatos(saida_dir)
    ia = {
        linha["candidato_id"]: linha
        for linha in conf.ler_tabela(
            saida_dir / julgar.RISCOS_IA_CSV,
            ["processo_id", "candidato_id", "categoria", "nomeia_objeto", "termo", "especifico_do_objeto", "grupo"],
        )
    }
    log: dict[str, dict[str, str]] = {}
    for linha in conf.ler_tabela(
        saida_dir / gerar.LOG_CSV, ["processo_id", "modelo", "versao_prompt", "data_hora", "situacao", "qtd_total"]
    ):
        if linha["situacao"] != "gerado":
            continue
        anterior = log.get(linha["processo_id"])
        if anterior is None or linha["data_hora"] >= anterior["data_hora"]:
            log[linha["processo_id"]] = linha

    problemas = []
    for pid in casos:
        if not riscos.get(pid):
            problemas.append(f"{pid}: sem riscos reais em {gerar.RISCOS_CSV}")
        if not candidatos.get(pid):
            problemas.append(f"{pid}: sem candidatos em {gerar.CANDIDATOS_CSV}")
            continue
        de_ia = [c["candidato_id"] for c in candidatos[pid] if c["origem"] == ORIGIN_AI]
        sem_linha = [cid for cid in de_ia if cid not in ia]
        if sem_linha:
            problemas.append(f"{pid}: {len(sem_linha)} risco(s) de IA sem linha em {julgar.RISCOS_IA_CSV} (ex.: {sem_linha[0]})")
        if pid not in log:
            problemas.append(f"{pid}: sem linha 'gerado' em {gerar.LOG_CSV}")
        elif log[pid]["qtd_total"] != str(len(de_ia)):
            problemas.append(
                f"{pid}: {gerar.LOG_CSV} registra {log[pid]['qtd_total']} risco(s) de IA e {gerar.CANDIDATOS_CSV} tem {len(de_ia)}"
            )
    if problemas:
        raise ConsolidacaoError("Entradas inconsistentes; nada foi gravado:\n" + "\n".join(f"  - {p}" for p in problemas))
    origem = {c["candidato_id"]: c["origem"] for lista in candidatos.values() for c in lista}
    return Dados(casos, riscos, candidatos, ia, log, origem)


def conferir_julgamento(corpus: Path, dados: Dados, analise: dict) -> None:
    """O Opus e o painel julgaram todos os riscos reais, e ``painel_votos.csv`` confere com os tres arquivos."""
    saida_dir = corpus / gerar.SAIDA_DIR
    esperados = {(pid, r["risco_id"]) for pid in dados.casos for r in dados.riscos[pid]}
    faltando = sorted(esperados - set(analise["escopo"]))
    if faltando:
        raise ConsolidacaoError(
            f"{len(faltando)} risco(s) reais sem julgamento do Opus ou do painel (ex.: {faltando[0][0]} / {faltando[0][1]})."
        )
    for linha in conf.ler_tabela(saida_dir / julgar.JULGAMENTO_CSV, ["processo_id", "risco_id", "cobertura"]):
        chave = (linha["processo_id"], linha["risco_id"])
        if chave in esperados and linha["cobertura"] != classe(analise["votos"][painel.REFERENCIA][chave], dados.origem):
            raise ConsolidacaoError(
                f"{julgar.JULGAMENTO_CSV}: {chave[0]} / {chave[1]} com cobertura diferente da dos candidatos marcados."
            )
    colunas = ["processo_id", "risco_id", "candidato_id", painel.REFERENCIA, *analise["nomes"], painel.MAIORIA]
    caminho = saida_dir / julgar.PAINEL_DIR / painel.VOTOS_CSV
    gravados = {
        tuple(linha[c] for c in colunas) for linha in conf.ler_tabela(caminho, colunas) if linha["processo_id"] in dados.casos
    }
    calculados = {tuple(linha[c] for c in colunas) for linha in painel.linhas_de_votos(analise)}
    if gravados != calculados:
        raise ConsolidacaoError(
            f"{painel.VOTOS_CSV} nao confere com os arquivos do painel ({len(gravados ^ calculados)} linha(s) diferentes); "
            "rode scripts/retro_painel.py de novo."
        )


# --------------------------------------------------------------------------
# Contagens
# --------------------------------------------------------------------------


def classe(marcados: set[str] | None, origem: dict[str, str]) -> str:
    if marcados is None:
        return SEM
    return julgar.derivar_cobertura([origem[c] for c in marcados])


def marcados_no_processo(dados: Dados, pid: str, votos: dict) -> set[str]:
    return {c for r in dados.riscos[pid] for c in (votos[(pid, r["risco_id"])] or ())}


def grupos_de_ia(dados: Dados, pid: str, marcados: set[str]) -> tuple[set[str], set[str]]:
    """(grupos distintos de riscos de IA do processo, grupos sem nenhum risco com correspondente)."""
    de_ia = [c["candidato_id"] for c in dados.candidatos[pid] if c["origem"] == ORIGIN_AI]
    todos = {dados.ia[cid]["grupo"] for cid in de_ia}
    return todos, todos - {dados.ia[cid]["grupo"] for cid in de_ia if cid in marcados}


def contar_processo(dados: Dados, pid: str, votos: dict) -> dict[str, int]:
    """Contagens de um processo segundo um conjunto de votos (o Opus ou a maioria)."""
    classes = Counter(classe(votos[(pid, r["risco_id"])], dados.origem) for r in dados.riscos[pid])
    marcados = marcados_no_processo(dados, pid, votos)
    de_ia = [c["candidato_id"] for c in dados.candidatos[pid] if c["origem"] == ORIGIN_AI]
    grupos, grupos_sem = grupos_de_ia(dados, pid, marcados)
    return {
        "riscos_reais": len(dados.riscos[pid]),
        "so_catalogo": classes["catalogo"],
        "so_ia": classes["ia"],
        "ambos": classes["ambos"],
        "coberto": classes["catalogo"] + classes["ia"] + classes["ambos"],
        "nenhum": classes["nenhum"],
        "sem_julgamento": classes[SEM],
        "ia_gerados": len(de_ia),
        "ia_grupos": len(grupos),
        "ia_nomeia_objeto": sum(1 for cid in de_ia if conf._sim_nao(dados.ia[cid]["nomeia_objeto"]) == "sim"),
        "ia_especifico_do_objeto": sum(1 for cid in de_ia if conf._sim_nao(dados.ia[cid]["especifico_do_objeto"]) == "sim"),
        "ia_sem_correspondente": sum(1 for cid in de_ia if cid not in marcados),
        "ia_grupos_sem_correspondente": len(grupos_sem),
    }


def somar(contagens: list[dict[str, int]]) -> dict[str, int]:
    return {campo: sum(c[campo] for c in contagens) for campo in SOMAVEIS}


def quartis(valores: list[float]) -> tuple[float, float, float] | None:
    """(Q1, mediana, Q3), com interpolacao linear entre os pontos (metodo "inclusivo")."""
    if not valores:
        return None
    if len(valores) == 1:
        return valores[0], valores[0], valores[0]
    q1, mediana, q3 = statistics.quantiles(valores, n=4, method="inclusive")
    return q1, mediana, q3


def _razao(parte: int, total: int) -> float | None:
    return parte / total if total else None


def medidas_por_processo(contagem: dict[str, int]) -> dict[str, float | None]:
    """As proporcoes de um processo que entram nas medianas."""
    reais, ia = contagem["riscos_reais"], contagem["ia_gerados"]
    return {
        "cobertura do catalogo": _razao(contagem["so_catalogo"] + contagem["ambos"], reais),
        "cobertura incremental da IA": _razao(contagem["so_ia"], reais),
        "cobertura total": _razao(contagem["coberto"], reais),
        "riscos de IA sem correspondente": _razao(contagem["ia_sem_correspondente"], ia),
        "riscos de IA com nomeia_objeto=sim": _razao(contagem["ia_nomeia_objeto"], ia),
        "riscos de IA com especifico_do_objeto=sim": _razao(contagem["ia_especifico_do_objeto"], ia),
        "grupos distintos de riscos de IA": float(contagem["ia_grupos"]),
    }


# --------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------


def _dec(valor: float | None, casas: int = 4) -> str:
    return "" if valor is None else f"{valor:.{casas}f}".replace(".", ",")


def _pct(parte: int, total: int) -> str:
    return _dec(100 * parte / total, 1) if total else ""


def _com_percentuais(contagem: dict[str, int]) -> dict:
    reais, ia = contagem["riscos_reais"], contagem["ia_gerados"]
    return {
        **contagem,
        "catalogo_com_ou_sem_ia": contagem["so_catalogo"] + contagem["ambos"],
        "pct_catalogo_com_ou_sem_ia": _pct(contagem["so_catalogo"] + contagem["ambos"], reais),
        "pct_so_ia": _pct(contagem["so_ia"], reais),
        "pct_coberto": _pct(contagem["coberto"], reais),
        "pct_nenhum": _pct(contagem["nenhum"], reais),
        "pct_ia_nomeia_objeto": _pct(contagem["ia_nomeia_objeto"], ia),
        "pct_ia_especifico_do_objeto": _pct(contagem["ia_especifico_do_objeto"], ia),
        "pct_ia_sem_correspondente": _pct(contagem["ia_sem_correspondente"], ia),
        "pct_ia_grupos_sem_correspondente": _pct(contagem["ia_grupos_sem_correspondente"], contagem["ia_grupos"]),
    }


CONTAGEM_FIELDS = [
    "riscos_reais", "so_catalogo", "so_ia", "ambos", "nenhum", "coberto", "catalogo_com_ou_sem_ia",
    "pct_catalogo_com_ou_sem_ia", "pct_so_ia", "pct_coberto", "pct_nenhum",
    "ia_gerados", "ia_grupos", "ia_nomeia_objeto", "pct_ia_nomeia_objeto",
    "ia_especifico_do_objeto", "pct_ia_especifico_do_objeto",
    "ia_sem_correspondente", "pct_ia_sem_correspondente",
    "ia_grupos_sem_correspondente", "pct_ia_grupos_sem_correspondente",
]  # fmt: skip


def _texto_do_candidato(candidato: dict[str, str], prefixo: str) -> dict[str, str]:
    return {
        f"{prefixo}_id": candidato["candidato_id"],
        f"{prefixo}_categoria": candidato["categoria"],
        f"{prefixo}_titulo": candidato["titulo"],
        f"{prefixo}_causa": candidato["causa"],
        f"{prefixo}_consequencia": candidato["consequencia"],
    }


def _campos_de_ia(dados: Dados, cid: str) -> dict[str, str]:
    linha = dados.ia[cid]
    return {
        "nomeia_objeto": linha["nomeia_objeto"],
        "termo": linha["termo"],
        "especifico_do_objeto": linha["especifico_do_objeto"],
        "grupo_ia": linha["grupo"],
    }


def _lista(candidatos: set[str] | None, ordem: list[str]) -> str:
    if candidatos is None:
        return SEM
    return julgar.SEPARADOR.join(cid for cid in ordem if cid in candidatos)


def consolidar_grupo(dados: Dados, grupo: str, pids: list[str], analise: dict) -> dict:
    """Todas as tabelas e medidas de um grupo de processos; ``analise`` e a do painel para esses processos."""
    nomes = analise["nomes"]
    votos = analise["votos"]
    opus, maioria = votos[painel.REFERENCIA], votos[painel.MAIORIA]
    por_opus = {pid: contar_processo(dados, pid, opus) for pid in pids}
    por_maioria = {pid: contar_processo(dados, pid, maioria) for pid in pids}
    tabelas: dict[str, list[dict]] = {}

    def base(pid: str) -> dict[str, str]:
        return {"grupo": grupo, "processo_id": pid, "tipo": dados.casos[pid]["tipo"]}

    tabelas["por_processo"] = [
        {
            **base(pid),
            **_com_percentuais(por_opus[pid]),
            "modelo_gerador": dados.log[pid]["modelo"],
            "versao_prompt_geracao": dados.log[pid]["versao_prompt"],
        }
        for pid in pids
    ]

    tipos = [tipo for tipo in gerar.TIPOS if any(dados.casos[pid]["tipo"] == tipo for pid in pids)]
    por_tipo, por_tipo_maioria = {}, {}
    for tipo in tipos:
        do_tipo = [pid for pid in pids if dados.casos[pid]["tipo"] == tipo]
        por_tipo[tipo] = {"processos": len(do_tipo), **somar([por_opus[pid] for pid in do_tipo])}
        por_tipo_maioria[tipo] = somar([por_maioria[pid] for pid in do_tipo])
    tabelas["por_tipo"] = [
        {"grupo": grupo, "tipo": tipo, "processos": por_tipo[tipo]["processos"], **_com_percentuais(por_tipo[tipo])}
        for tipo in tipos
    ]

    categoria = {c["candidato_id"]: c["categoria"] for pid in pids for c in dados.candidatos[pid]}
    vistas = sorted({categoria[cid] for cid in categoria} - set(CATEGORIES))
    por_categoria = []
    total_de_riscos = sum(len(dados.riscos[pid]) for pid in pids)
    marcados = {c for pid in pids for c in marcados_no_processo(dados, pid, opus)}
    for cat in [*CATEGORIES, *vistas]:
        do_catalogo = [cid for cid, valor in categoria.items() if valor == cat and dados.origem[cid] == ORIGIN_CATALOG]
        de_ia = [cid for cid, valor in categoria.items() if valor == cat and dados.origem[cid] == ORIGIN_AI]
        if not do_catalogo and not de_ia:
            continue
        cobertos = Counter()
        for pid in pids:
            for risco in dados.riscos[pid]:
                origens = {dados.origem[c] for c in opus[(pid, risco["risco_id"])] if categoria[c] == cat}
                cobertos["catalogo"] += ORIGIN_CATALOG in origens
                cobertos["ia"] += ORIGIN_AI in origens
                cobertos["qualquer"] += bool(origens)
        sem = [cid for cid in de_ia if cid not in marcados]
        nomeia = sum(1 for cid in de_ia if conf._sim_nao(dados.ia[cid]["nomeia_objeto"]) == "sim")
        especifico = sum(1 for cid in de_ia if conf._sim_nao(dados.ia[cid]["especifico_do_objeto"]) == "sim")
        por_categoria.append(
            {
                "grupo": grupo,
                "categoria": cat,
                "candidatos_catalogo": len(do_catalogo),
                "riscos_reais_cobertos_pelo_catalogo": cobertos["catalogo"],
                "riscos_reais_cobertos_pela_ia": cobertos["ia"],
                "riscos_reais_cobertos": cobertos["qualquer"],
                "pct_riscos_reais_cobertos": _pct(cobertos["qualquer"], total_de_riscos),
                "ia_gerados": len(de_ia),
                "ia_sem_correspondente": len(sem),
                "pct_ia_sem_correspondente": _pct(len(sem), len(de_ia)),
                "ia_nomeia_objeto": nomeia,
                "pct_ia_nomeia_objeto": _pct(nomeia, len(de_ia)),
                "ia_especifico_do_objeto": especifico,
                "pct_ia_especifico_do_objeto": _pct(especifico, len(de_ia)),
            }
        )
    tabelas["por_categoria"] = por_categoria

    nao_cobertos, incremental, sem_correspondente, divergencias, robustez_processo = [], [], [], [], []
    for pid in pids:
        ordem = [c["candidato_id"] for c in dados.candidatos[pid]]
        candidato = {c["candidato_id"]: c for c in dados.candidatos[pid]}
        marcados_opus = marcados_no_processo(dados, pid, opus)
        marcados_maioria = marcados_no_processo(dados, pid, maioria)
        _, grupos_sem = grupos_de_ia(dados, pid, marcados_opus)
        mesmo_resultado = 0
        for risco in dados.riscos[pid]:
            chave = (pid, risco["risco_id"])
            classe_opus, classe_maioria = classe(opus[chave], dados.origem), classe(maioria[chave], dados.origem)
            texto = {
                **base(pid),
                "risco_id": risco["risco_id"],
                "fase": risco.get("fase", ""),
                "risco": risco["risco"],
                "causa": risco["causa"],
                "consequencia": risco["consequencia"],
            }
            if classe_opus == "nenhum":
                nao_cobertos.append({**texto, "cobertura_pela_maioria": classe_maioria})
            if classe_opus == "ia":
                for cid in ordem:
                    if cid in opus[chave]:
                        incremental.append({**texto, **_texto_do_candidato(candidato[cid], "ia"), **_campos_de_ia(dados, cid)})
            if classe_maioria != SEM and (classe_opus == "nenhum") == (classe_maioria == "nenhum"):
                mesmo_resultado += 1
            elif classe_maioria != SEM:
                citados = [cid for cid in ordem if any(cid in (votos[n][chave] or ()) for n in (painel.REFERENCIA, *nomes))]
                divergencias.append(
                    {
                        **texto,
                        "cobertura_opus": classe_opus,
                        "cobertura_maioria": classe_maioria,
                        "candidatos_opus": _lista(opus[chave], ordem),
                        "candidatos_maioria": _lista(maioria[chave], ordem),
                        **{f"candidatos_{nome}": _lista(votos[nome][chave], ordem) for nome in nomes},
                        **{f"cobertura_{nome}": classe(votos[nome][chave], dados.origem) for nome in nomes},
                        "titulos_dos_candidatos": julgar.SEPARADOR.join(f"{cid}: {candidato[cid]['titulo']}" for cid in citados),
                    }
                )
        for cid in ordem:
            if dados.origem[cid] == ORIGIN_AI and cid not in marcados_opus:
                campos = _campos_de_ia(dados, cid)
                sem_correspondente.append(
                    {
                        **base(pid),
                        "candidato_id": cid,
                        "categoria": candidato[cid]["categoria"],
                        "titulo": candidato[cid]["titulo"],
                        "causa": candidato[cid]["causa"],
                        "consequencia": candidato[cid]["consequencia"],
                        **campos,
                        "grupo_ia_sem_correspondente": "sim" if campos["grupo_ia"] in grupos_sem else "nao",
                        "sem_correspondente_pela_maioria": "nao" if cid in marcados_maioria else "sim",
                    }
                )
        o, m = por_opus[pid], por_maioria[pid]
        robustez_processo.append(
            {
                **base(pid),
                "riscos_reais": o["riscos_reais"],
                "coberto_opus": o["coberto"],
                "pct_coberto_opus": _pct(o["coberto"], o["riscos_reais"]),
                "coberto_maioria": m["coberto"],
                "pct_coberto_maioria": _pct(m["coberto"], m["riscos_reais"]),
                "sem_maioria": m["sem_julgamento"],
                "riscos_com_mesmo_resultado": mesmo_resultado,
                "ia_gerados": o["ia_gerados"],
                "ia_sem_correspondente_opus": o["ia_sem_correspondente"],
                "pct_ia_sem_correspondente_opus": _pct(o["ia_sem_correspondente"], o["ia_gerados"]),
                "ia_sem_correspondente_maioria": m["ia_sem_correspondente"],
                "pct_ia_sem_correspondente_maioria": _pct(m["ia_sem_correspondente"], m["ia_gerados"]),
            }
        )
    tabelas["nao_cobertos"] = nao_cobertos
    tabelas["ia_cobertura_incremental"] = incremental
    tabelas["ia_sem_correspondente"] = sem_correspondente
    tabelas["divergencias_maioria_opus"] = divergencias
    tabelas["robustez_por_processo"] = robustez_processo

    escopo = len(analise["escopo"])
    robustez_cobertura = []
    for nome in (painel.REFERENCIA, *nomes, painel.MAIORIA):
        dados_do_juiz = analise["por_juiz"][nome]
        classes = dados_do_juiz["classes"]
        coberto = classes["catalogo"] + classes["ia"] + classes["ambos"]
        robustez_cobertura.append(
            {
                "grupo": grupo,
                "juiz": nome,
                "modelo": juizes.JUIZES[nome].modelo if nome in juizes.JUIZES else "",
                "riscos_reais": escopo,
                "so_catalogo": classes["catalogo"],
                "so_ia": classes["ia"],
                "ambos": classes["ambos"],
                "coberto": coberto,
                "nenhum": classes["nenhum"],
                "sem_julgamento": classes[SEM],
                "pct_so_catalogo": _pct(classes["catalogo"], escopo),
                "pct_so_ia": _pct(classes["ia"], escopo),
                "pct_ambos": _pct(classes["ambos"], escopo),
                "pct_coberto": _pct(coberto, escopo),
                "pct_nenhum": _pct(classes["nenhum"], escopo),
                "candidatos_marcados": dados_do_juiz["marcados"],
            }
        )
    tabelas["robustez_cobertura"] = robustez_cobertura

    concordancia = [
        {
            "grupo": grupo,
            "medida": "kappa de Cohen por par",
            "a": a,
            "b": b,
            "unidade": "par (risco real, candidato)",
            "n": comparacao["pares"],
            "concordancia": _dec(comparacao["concordancia"]),
            "kappa": _dec(comparacao["kappa"]),
        }
        for (a, b), comparacao in analise["comparacoes"].items()
    ]
    fleiss, mxr = analise["fleiss"], analise["maioria_x_referencia"]
    concordancia.append(
        {
            "grupo": grupo,
            "medida": "kappa de Fleiss por par",
            "a": " + ".join(nomes),
            "unidade": "par (risco real, candidato)",
            "n": fleiss["pares"],
            "kappa": _dec(fleiss["kappa"]),
        }
    )
    for medida, campo in (("mesmo resultado coberto/nao coberto", "mesmo_coberto"), ("mesma classe de cobertura", "mesma_classe")):
        concordancia.append(
            {
                "grupo": grupo,
                "medida": medida,
                "a": painel.MAIORIA,
                "b": painel.REFERENCIA,
                "unidade": "risco real",
                "n": mxr["riscos"],
                "iguais": mxr[campo],
                "concordancia": _dec(_razao(mxr[campo], mxr["riscos"])),
            }
        )
    tabelas["robustez_concordancia"] = concordancia

    total_opus, total_maioria = somar(list(por_opus.values())), somar(list(por_maioria.values()))
    tabelas["robustez_ia_sem_correspondente"] = [
        {
            "grupo": grupo,
            "ia_gerados": total_opus["ia_gerados"],
            "ia_sem_correspondente_opus": total_opus["ia_sem_correspondente"],
            "pct_ia_sem_correspondente_opus": _pct(total_opus["ia_sem_correspondente"], total_opus["ia_gerados"]),
            "ia_sem_correspondente_maioria": total_maioria["ia_sem_correspondente"],
            "pct_ia_sem_correspondente_maioria": _pct(total_maioria["ia_sem_correspondente"], total_maioria["ia_gerados"]),
            "ia_grupos": total_opus["ia_grupos"],
            "ia_grupos_sem_correspondente_opus": total_opus["ia_grupos_sem_correspondente"],
            "pct_ia_grupos_sem_correspondente_opus": _pct(total_opus["ia_grupos_sem_correspondente"], total_opus["ia_grupos"]),
            "ia_grupos_sem_correspondente_maioria": total_maioria["ia_grupos_sem_correspondente"],
            "pct_ia_grupos_sem_correspondente_maioria": _pct(
                total_maioria["ia_grupos_sem_correspondente"], total_maioria["ia_grupos"]
            ),
        }
    ]

    def medianas(por_processo: dict[str, dict[str, int]]) -> dict[str, tuple | None]:
        medidas = [medidas_por_processo(por_processo[pid]) for pid in pids]
        return {nome: quartis([m[nome] for m in medidas if m[nome] is not None]) for nome in medidas[0]}

    return {
        "grupo": grupo,
        "processos": len(pids),
        "tabelas": tabelas,
        "total": {painel.REFERENCIA: total_opus, painel.MAIORIA: total_maioria},
        "por_tipo": {painel.REFERENCIA: por_tipo, painel.MAIORIA: por_tipo_maioria},
        "medianas": {painel.REFERENCIA: medianas(por_opus), painel.MAIORIA: medianas(por_maioria)},
        "analise": analise,
    }


CAMPOS = {
    "por_processo": ["grupo", "processo_id", "tipo", *CONTAGEM_FIELDS, "modelo_gerador", "versao_prompt_geracao"],
    "por_tipo": ["grupo", "tipo", "processos", *CONTAGEM_FIELDS],
    "por_categoria": [
        "grupo", "categoria", "candidatos_catalogo", "riscos_reais_cobertos_pelo_catalogo", "riscos_reais_cobertos_pela_ia",
        "riscos_reais_cobertos", "pct_riscos_reais_cobertos", "ia_gerados", "ia_sem_correspondente",
        "pct_ia_sem_correspondente", "ia_nomeia_objeto", "pct_ia_nomeia_objeto", "ia_especifico_do_objeto",
        "pct_ia_especifico_do_objeto",
    ],
    "nao_cobertos": [
        "grupo", "processo_id", "tipo", "risco_id", "fase", "risco", "causa", "consequencia", "cobertura_pela_maioria",
    ],
    "ia_sem_correspondente": [
        "grupo", "processo_id", "tipo", "candidato_id", "categoria", "titulo", "causa", "consequencia", "nomeia_objeto",
        "termo", "especifico_do_objeto", "grupo_ia", "grupo_ia_sem_correspondente", "sem_correspondente_pela_maioria",
    ],
    "ia_cobertura_incremental": [
        "grupo", "processo_id", "tipo", "risco_id", "fase", "risco", "causa", "consequencia", "ia_id", "ia_categoria",
        "ia_titulo", "ia_causa", "ia_consequencia", "nomeia_objeto", "termo", "especifico_do_objeto", "grupo_ia",
    ],
    "robustez_cobertura": [
        "grupo", "juiz", "modelo", "riscos_reais", "so_catalogo", "so_ia", "ambos", "coberto", "nenhum", "sem_julgamento",
        "pct_so_catalogo", "pct_so_ia", "pct_ambos", "pct_coberto", "pct_nenhum", "candidatos_marcados",
    ],
    "robustez_concordancia": ["grupo", "medida", "a", "b", "unidade", "n", "iguais", "concordancia", "kappa"],
    "robustez_por_processo": [
        "grupo", "processo_id", "tipo", "riscos_reais", "coberto_opus", "pct_coberto_opus", "coberto_maioria",
        "pct_coberto_maioria", "sem_maioria", "riscos_com_mesmo_resultado", "ia_gerados", "ia_sem_correspondente_opus",
        "pct_ia_sem_correspondente_opus", "ia_sem_correspondente_maioria", "pct_ia_sem_correspondente_maioria",
    ],
    "robustez_ia_sem_correspondente": [
        "grupo", "ia_gerados", "ia_sem_correspondente_opus", "pct_ia_sem_correspondente_opus",
        "ia_sem_correspondente_maioria", "pct_ia_sem_correspondente_maioria", "ia_grupos",
        "ia_grupos_sem_correspondente_opus", "pct_ia_grupos_sem_correspondente_opus",
        "ia_grupos_sem_correspondente_maioria", "pct_ia_grupos_sem_correspondente_maioria",
    ],
}  # fmt: skip


def _campos_de_divergencias(nomes: list[str]) -> list[str]:
    return [
        "grupo", "processo_id", "tipo", "risco_id", "fase", "risco", "causa", "consequencia", "cobertura_opus",
        "cobertura_maioria", *(f"cobertura_{nome}" for nome in nomes), "candidatos_opus", "candidatos_maioria",
        *(f"candidatos_{nome}" for nome in nomes), "titulos_dos_candidatos",
    ]  # fmt: skip


# --------------------------------------------------------------------------
# Consolidacao e relatorio
# --------------------------------------------------------------------------


def consolidar(corpus: Path, nomes: list[str] | None = None) -> dict[str, list[dict]]:
    """``{conjunto: [resultado de cada grupo]}``; nao grava nada."""
    corpus = corpus.resolve()
    dados = carregar(corpus)
    conferir_julgamento(corpus, dados, painel.analisar(corpus, nomes, set(dados.casos)))
    resultado: dict[str, list[dict]] = {}
    for conjunto in CONJUNTOS:
        resultado[conjunto] = []
        for grupo in dict.fromkeys(caso["grupo"] for caso in dados.casos.values()):
            pids = [
                pid
                for pid, caso in dados.casos.items()
                if caso["grupo"] == grupo and (conjunto == "sensibilidade" or caso["principal"] == "sim")
            ]
            if pids:
                resultado[conjunto].append(consolidar_grupo(dados, grupo, pids, painel.analisar(corpus, nomes, set(pids))))
    return resultado


def gravar(corpus: Path, resultado: dict[str, list[dict]]) -> list[Path]:
    """Grava as tabelas de cada conjunto; os grupos ficam em linhas separadas, pela coluna ``grupo``."""
    gravados = []
    for conjunto, grupos in resultado.items():
        if not grupos:
            continue
        pasta = corpus.resolve() / gerar.SAIDA_DIR / RESULTADOS_DIR / conjunto
        pasta.mkdir(parents=True, exist_ok=True)
        campos = {**CAMPOS, "divergencias_maioria_opus": _campos_de_divergencias(grupos[0]["analise"]["nomes"])}
        for tabela, colunas in campos.items():
            linhas = [linha for grupo in grupos for linha in grupo["tabelas"][tabela]]
            conf.gravar_tabela(pasta / f"{tabela}.csv", colunas, linhas)
            gravados.append(pasta / f"{tabela}.csv")
    return gravados


def _n(valor: float | None, casas: int = 3) -> str:
    return "n/d" if valor is None else f"{valor:.{casas}f}".replace(".", ",")


def _parte(parte: int, total: int) -> str:
    return f"{parte} ({_n(100 * parte / total, 1)}%)" if total else f"{parte}"


def _cobertura(rotulo: str, c: dict[str, int], largura: int = 26) -> str:
    reais = c["riscos_reais"]
    valores = [c["so_catalogo"], c["so_ia"], c["ambos"], c["coberto"], c["nenhum"], c["so_catalogo"] + c["ambos"]]
    return f"{rotulo:<{largura}}{reais:>8}" + "".join(f"{_parte(valor, reais):>14}" for valor in valores)


_CABECALHO_COBERTURA = "".join(f"{c:>14}" for c in ("so catalogo", "so IA", "ambos", "total", "nenhum", "catalogo*"))


def relatorio(resultado: dict[str, list[dict]]) -> list[str]:
    titulos = {
        "principal": "ANALISE PRINCIPAL (incluir=sim e analise_principal=sim)",
        "sensibilidade": "SENSIBILIDADE (todos com incluir=sim)",
    }
    linhas: list[str] = []
    for conjunto, grupos in resultado.items():
        for g in grupos:
            opus, maioria = g["total"][painel.REFERENCIA], g["total"][painel.MAIORIA]
            linhas += [
                "",
                f"######## {titulos[conjunto]} | grupo {g['grupo']}: {g['processos']} processo(s), "
                f"{opus['riscos_reais']} riscos reais, {opus['ia_gerados']} riscos de IA ########",
                "",
                "== Cobertura dos riscos reais ==",
                f"{'julgamento':<26}{'riscos':>8}" + _CABECALHO_COBERTURA,
                _cobertura("Opus (principal)", opus),
                _cobertura("maioria do painel", maioria),
                "(* catalogo, com ou sem IA = so catalogo + ambos; total = so catalogo + so IA + ambos)",
            ]
            if maioria["sem_julgamento"]:
                linhas.append(f"Riscos sem maioria: {maioria['sem_julgamento']}")

            ia = opus["ia_gerados"]
            linhas += [
                "",
                "== Riscos de IA ==",
                f"Gerados: {ia}, em {opus['ia_grupos']} grupo(s) distintos (somados por processo)",
                f"Com nomeia_objeto=sim: {_parte(opus['ia_nomeia_objeto'], ia)}",
                f"Com especifico_do_objeto=sim: {_parte(opus['ia_especifico_do_objeto'], ia)}",
                f"Sem correspondente no mapa real: Opus {_parte(opus['ia_sem_correspondente'], ia)}; "
                f"maioria {_parte(maioria['ia_sem_correspondente'], ia)}",
                f"Grupos sem correspondente: Opus {_parte(opus['ia_grupos_sem_correspondente'], opus['ia_grupos'])}; "
                f"maioria {_parte(maioria['ia_grupos_sem_correspondente'], maioria['ia_grupos'])}",
            ]

            linhas += ["", "== Por processo: mediana [Q1 - Q3] ==", f"{'medida':<44}{'Opus':>26}{'maioria':>26}"]
            independem = ("nomeia_objeto", "especifico_do_objeto", "grupos distintos")
            for nome, de_opus in g["medianas"][painel.REFERENCIA].items():
                em_pct = not nome.startswith("grupos")

                def texto(q: tuple | None) -> str:
                    if q is None:
                        return "n/d"
                    q1, mediana, q3 = (100 * v if em_pct else v for v in q)
                    sufixo = "%" if em_pct else ""
                    return f"{_n(mediana, 1)}{sufixo} [{_n(q1, 1)} - {_n(q3, 1)}]"

                de_maioria = "" if any(parte in nome for parte in independem) else texto(g["medianas"][painel.MAIORIA][nome])
                linhas.append(f"{nome:<44}{texto(de_opus):>26}{de_maioria:>26}")
            linhas.append("(quartis com interpolacao linear; as tres ultimas medidas nao dependem do julgamento de cobertura)")

            linhas += ["", "== Cobertura por tipo de contratacao ==", f"{'tipo (processos)':<26}{'riscos':>8}" + _CABECALHO_COBERTURA]
            for tipo, contagem in g["por_tipo"][painel.REFERENCIA].items():
                rotulo = f"{tipo} ({contagem['processos']})"
                linhas.append(_cobertura(f"{rotulo} Opus", contagem))
                linhas.append(_cobertura(f"{'':<{len(rotulo)}} maioria", g["por_tipo"][painel.MAIORIA][tipo]))

            analise = g["analise"]
            mxr, geral = analise["maioria_x_referencia"], analise["comparacoes"][(painel.MAIORIA, painel.REFERENCIA)]
            linhas += [
                "",
                "== Robustez: maioria do painel contra o Opus ==",
                f"Kappa de Cohen por par: {_n(geral['kappa'])} (concordancia {_n(geral['concordancia'])}, {geral['pares']} pares)",
                f"Kappa de Fleiss dos {len(analise['nomes'])} juizes: {_n(analise['fleiss']['kappa'])}",
                f"Mesmo resultado coberto/nao coberto: {_parte(mxr['mesmo_coberto'], mxr['riscos'])} de {mxr['riscos']} riscos",
                f"Mesma classe de cobertura: {_parte(mxr['mesma_classe'], mxr['riscos'])} de {mxr['riscos']} riscos",
            ]
    return linhas[1:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", required=True, type=Path, help="Pasta do corpus.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"Pasta do corpus nao encontrada: {corpus}", file=sys.stderr)
        return 2
    try:
        resultado = consolidar(corpus)
        print("\n".join(relatorio(resultado)))
        gravados = gravar(corpus, resultado)
    except (
        ConsolidacaoError, painel.PainelError, julgar.JulgamentoError, conf.ConferenciaError, gerar.GeracaoError
    ) as exc:  # fmt: skip
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    print(f"\n{len(gravados)} arquivo(s) gravado(s) em {corpus / gerar.SAIDA_DIR / RESULTADOS_DIR}:")
    for caminho in gravados:
        print(f"  {caminho.parent.name}\\{caminho.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
