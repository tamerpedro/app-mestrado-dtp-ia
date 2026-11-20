"""Avaliacao retrospectiva, Parte 0b: conferencia manual da extracao e aplicacao das correcoes.

Nao chama modelo de IA e nunca regrava ``corpus_casos.csv``. Dois modos, sempre
sobre a pasta do corpus (``--corpus``):

``--gerar`` grava ``conferencia_extracao.csv`` com uma linha por campo a
conferir contra o PDF:

- enunciado de todo risco com ``conferir=sim`` (ou que a checagem literal
  atual nao acha na pagina indicada);
- causa ou consequencia que a mesma checagem literal nao acha na pagina
  indicada (a consequencia e conferida dano a dano; basta um nao achado);
- amostra aleatoria dos riscos restantes (15%, minimo 30, ao menos um por
  processo, semente registrada na ultima linha do arquivo), com uma linha por
  campo preenchido.

Onde a camada de texto do PDF perde a ligadura "ti" (caractere nulo no lugar
dela), ``texto_sugerido`` traz o texto restaurado; nada e aplicado sozinho.

``--aplicar`` le o arquivo preenchido pelo autor (``ok_autor`` = sim/nao,
``texto_corrigido``), guarda a extracao intocada em
``corpus_riscos_reais_original.csv`` na primeira execucao, altera so os campos
corrigidos em ``corpus_riscos_reais.csv`` e registra cada alteracao em
``correcoes_extracao.csv``. Pode ser executado de novo sem duplicar nada.
``texto_corrigido`` igual a ``[VAZIO]`` limpa o campo (o conteudo extraido nao
pertence a ele). Alem das linhas da conferencia, aplica uma regra fixa: nos
processos de ``NIVEL_COMO_CONSEQUENCIA`` (IBRAM-01), a consequencia que e so um
nivel de impacto ("Alto", "Medio", "Baixo", "Muito Alto") e limpa.

Uso:
    python scripts/retro_conferencia.py --corpus "<pasta>" --gerar [--semente N] [--sobrescrever]
    python scripts/retro_conferencia.py --corpus "<pasta>" --aplicar
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import retro_extrair as extr  # noqa: E402

CONFERENCIA_CSV = "conferencia_extracao.csv"
ORIGINAL_CSV = "corpus_riscos_reais_original.csv"
CORRECOES_CSV = "correcoes_extracao.csv"

CONFERENCIA_FIELDS = [
    "processo_id",
    "risco_id",
    "pagina",
    "campo",
    "texto_extraido",
    "texto_sugerido",
    "motivo",
    "ok_autor",
    "texto_corrigido",
    "observacao",
]
CORRECOES_FIELDS = ["processo_id", "risco_id", "campo", "texto_antes", "texto_depois", "motivo", "data_hora"]
CAMPOS = ("risco", "causa", "consequencia")

MOTIVO_ENUNCIADO = "enunciado nao encontrado"
MOTIVO_CAUSA = "causa nao encontrada"
MOTIVO_CONSEQUENCIA = "consequencia nao encontrada"
MOTIVO_AMOSTRA = "amostra"
MOTIVO_NIVEL = "nivel de impacto gravado como consequencia"

VAZIO = "[VAZIO]"  # em texto_corrigido: a correcao e limpar o campo
NIVEL_COMO_CONSEQUENCIA = {"IBRAM-01"}  # o documento traz o nivel no campo "Impactos"
NIVEIS_DE_IMPACTO = {"alto", "medio", "baixo", "muito alto"}
ERRO_TRANSCRICAO = "transcricao"  # o texto do campo estava errado e foi reescrito
ERRO_CAMPO_TROCADO = "campo trocado"  # o conteudo nao pertencia ao campo e foi limpo

SEMENTE_PADRAO = 20261117
FRACAO_AMOSTRA = 0.15
MINIMO_AMOSTRA = 30
MARCA_METADADO = "#"  # linha de metadado (semente); ignorada pelo --aplicar
SEPARADOR_DANOS = " | "
NULO = "\x00"  # o que sobra, na camada de texto, no lugar da ligadura "ti"


class ConferenciaError(Exception):
    """Falha que impede gerar ou aplicar a conferencia."""


# --------------------------------------------------------------------------
# Leitura tolerante (arquivos editados no Excel) e gravacao
# --------------------------------------------------------------------------


def _decodificar(dados: bytes) -> str:
    if dados.startswith((b"\xff\xfe", b"\xfe\xff")):  # "Texto Unicode" do Excel
        return dados.decode("utf-16")
    return extr._decodificar(dados)


def ler_tabela(caminho: Path, obrigatorias: list[str]) -> list[dict[str, str]]:
    """CSV com ``;`` ou tabulacao, em UTF-8 (com ou sem BOM), UTF-16 ou Windows-1252."""
    if not caminho.is_file():
        raise ConferenciaError(f"Arquivo nao encontrado: {caminho}")
    texto = _decodificar(caminho.read_bytes())
    primeira = texto.split("\n", 1)[0]
    leitor = csv.reader(io.StringIO(texto, newline=""), delimiter=extr._detectar_separador(primeira))
    try:
        cabecalho = [coluna.strip().lower() for coluna in next(leitor)]
    except StopIteration:
        raise ConferenciaError(f"{caminho.name} esta vazio.") from None
    faltando = [coluna for coluna in obrigatorias if coluna not in cabecalho]
    if faltando:
        raise ConferenciaError(f"{caminho.name} sem as colunas: {', '.join(faltando)}")
    registros = []
    for valores in leitor:
        if not any(valor.strip() for valor in valores):
            continue
        valores = valores + [""] * (len(cabecalho) - len(valores))
        registros.append({coluna: valor.strip() for coluna, valor in zip(cabecalho, valores)})
    return registros


def gravar_tabela(caminho: Path, campos: list[str], linhas: list[dict], *, acrescentar: bool = False) -> None:
    """CSV com ``;`` e UTF-8 com BOM. Sem ``acrescentar``, troca o arquivo de uma vez."""
    try:
        if acrescentar and caminho.is_file():
            with caminho.open("a", encoding="utf-8", newline="") as arquivo:
                csv.DictWriter(arquivo, fieldnames=campos, delimiter=";", extrasaction="ignore").writerows(linhas)
            return
        temporario = caminho.with_name(caminho.name + ".tmp")
        with temporario.open("w", encoding="utf-8-sig", newline="") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";", extrasaction="ignore")
            escritor.writeheader()
            escritor.writerows(linhas)
        os.replace(temporario, caminho)
    except PermissionError as exc:
        raise ConferenciaError(f"Nao foi possivel gravar {caminho.name} (aberto no Excel?): {exc}") from exc


def _uma_linha(texto: str) -> str:
    return " ".join((texto or "").split())


def _sim_nao(valor: str) -> str:
    return extr._sem_acento(valor).strip().casefold()


# --------------------------------------------------------------------------
# --gerar
# --------------------------------------------------------------------------


def formas_sem_ti(paginas: list[extr.Pagina]) -> list[str]:
    """Palavras do PDF com o caractere nulo no lugar de "ti", sem pontuacao nas pontas."""
    formas = {
        palavra.texto.strip(".,;:!?()[]\"'“”")
        for pagina in paginas
        for palavra in pagina.palavras
        if NULO in palavra.texto
    }
    return sorted((forma for forma in formas if forma.strip(NULO)), key=len, reverse=True)


def restaurar_ti(texto: str, formas: list[str]) -> str:
    """Repoe "ti" onde o texto extraido traz a lacuna de uma das ``formas`` do PDF."""
    restaurado = texto
    for forma in formas:
        certo = forma.replace(NULO, "ti")
        for lacuna in (" ", NULO):
            quebrado = re.escape(forma.replace(NULO, lacuna))
            restaurado = re.sub(rf"(?<!\w){quebrado}(?!\w)", lambda _m: certo, restaurado)
    return restaurado


def _achado(trecho: str, paginas: dict[int, extr.Pagina], numero: int | None) -> bool:
    pagina = paginas.get(numero) if numero is not None else None
    if pagina is None:
        return False
    return extr.localizar_trecho(trecho, pagina, paginas.get(numero + 1)) is not None


def _pendencias(risco: dict, paginas: dict[int, extr.Pagina]) -> list[tuple[str, str]]:
    """Campos do risco que precisam de conferencia obrigatoria, com o motivo."""
    numero = int(risco["pagina"]) if risco["pagina"].isdigit() else None
    pendencias = []
    if _sim_nao(risco["conferir"]) == "sim" or not _achado(risco["risco"], paginas, numero):
        pendencias.append(("risco", MOTIVO_ENUNCIADO))
    if risco["causa"] and not _achado(risco["causa"], paginas, numero):
        pendencias.append(("causa", MOTIVO_CAUSA))
    danos = [dano for dano in risco["consequencia"].split(SEPARADOR_DANOS) if dano.strip()]
    if any(not _achado(dano, paginas, numero) for dano in danos):
        pendencias.append(("consequencia", MOTIVO_CONSEQUENCIA))
    return pendencias


def sortear_amostra(restantes: list[dict], semente: int) -> list[dict]:
    """15% dos riscos restantes, no minimo 30 e ao menos um por processo."""
    tamanho = min(len(restantes), max(MINIMO_AMOSTRA, math.ceil(FRACAO_AMOSTRA * len(restantes))))
    sorteio = random.Random(semente)
    por_processo: dict[str, list[dict]] = defaultdict(list)
    for risco in restantes:
        por_processo[risco["processo_id"]].append(risco)
    escolhidos = [sorteio.choice(por_processo[processo]) for processo in sorted(por_processo)]
    ja = {id(risco) for risco in escolhidos}
    demais = [risco for risco in restantes if id(risco) not in ja]
    escolhidos += sorteio.sample(demais, max(0, tamanho - len(escolhidos)))
    return escolhidos


def gerar(
    corpus: Path,
    *,
    semente: int = SEMENTE_PADRAO,
    sobrescrever: bool = False,
    leitor: extr.Leitor = extr.extrair_paginas,
    saida: Callable[[str], None] = print,
) -> list[dict]:
    """Grava ``conferencia_extracao.csv`` e devolve as linhas (sem a de metadado)."""
    destino = corpus / CONFERENCIA_CSV
    if destino.exists() and not sobrescrever:
        raise ConferenciaError(
            f"{CONFERENCIA_CSV} ja existe (pode estar preenchido). Use --sobrescrever para gerar de novo."
        )
    riscos = ler_tabela(corpus / extr.RISCOS_CSV, extr.RISCOS_FIELDS)
    repetidos = [k for k, n in Counter((r["processo_id"], r["risco_id"]) for r in riscos).items() if n > 1]
    if repetidos:
        raise ConferenciaError(f"risco_id repetido no mesmo processo: {repetidos}")
    casos = {caso.processo_id: caso for caso in extr.ler_casos(corpus / extr.CASOS_CSV)}

    paginas: dict[str, dict[int, extr.Pagina]] = {}
    formas: dict[str, list[str]] = {}
    for processo in sorted({risco["processo_id"] for risco in riscos}):
        caso = casos.get(processo)
        if caso is None:
            raise ConferenciaError(f"{processo} esta em {extr.RISCOS_CSV} mas nao esta incluido em {extr.CASOS_CSV}.")
        lidas = leitor(corpus / extr.PDF_DIR / caso.arquivo, caso.pagina_inicial, caso.pagina_final)
        paginas[processo] = {pagina.numero: pagina for pagina in lidas}
        formas[processo] = formas_sem_ti(lidas)

    linhas: list[dict] = []

    def acrescentar(ordem: int, risco: dict, campo: str, motivo: str) -> None:
        texto = risco[campo]
        sugerido = restaurar_ti(texto, formas[risco["processo_id"]])
        linhas.append(
            {
                "processo_id": risco["processo_id"],
                "risco_id": risco["risco_id"],
                "pagina": risco["pagina"],
                "campo": campo,
                "texto_extraido": texto,
                "texto_sugerido": sugerido if sugerido != texto else "",
                "motivo": motivo,
                "ok_autor": "",
                "texto_corrigido": "",
                "observacao": "",
                "_ordem": (
                    risco["processo_id"],
                    int(risco["pagina"]) if risco["pagina"].isdigit() else 0,
                    ordem,
                    CAMPOS.index(campo),
                ),
            }
        )

    ordem_de = {id(risco): ordem for ordem, risco in enumerate(riscos)}
    restantes = []
    for risco in riscos:
        pendencias = _pendencias(risco, paginas[risco["processo_id"]])
        for campo, motivo in pendencias:
            acrescentar(ordem_de[id(risco)], risco, campo, motivo)
        if not pendencias:
            restantes.append(risco)
    amostra = sortear_amostra(restantes, semente)
    for risco in amostra:
        for campo in CAMPOS:
            if risco[campo]:
                acrescentar(ordem_de[id(risco)], risco, campo, MOTIVO_AMOSTRA)

    linhas.sort(key=lambda linha: linha["_ordem"])
    metadado = {
        "processo_id": MARCA_METADADO,
        "campo": "semente",
        "texto_extraido": str(semente),
        "motivo": (
            f"metadado: amostra de {len(amostra)} entre {len(restantes)} riscos restantes "
            f"({FRACAO_AMOSTRA:.0%}, minimo {MINIMO_AMOSTRA}, ao menos um por processo); nao preencher"
        ),
    }
    gravar_tabela(destino, CONFERENCIA_FIELDS, [*linhas, metadado])

    por_motivo = Counter(linha["motivo"] for linha in linhas)
    saida(f"{CONFERENCIA_CSV}: {len(linhas)} linha(s) a conferir; semente {semente}.")
    saida(f"Amostra: {len(amostra)} risco(s) entre {len(restantes)} restantes (de {len(riscos)} no total).")
    for motivo in (MOTIVO_ENUNCIADO, MOTIVO_CAUSA, MOTIVO_CONSEQUENCIA, MOTIVO_AMOSTRA):
        saida(f"  {motivo}: {por_motivo[motivo]}")
    return linhas


# --------------------------------------------------------------------------
# --aplicar
# --------------------------------------------------------------------------


def _so_nivel(processo_id: str, campo: str, texto: str) -> bool:
    """A regra do nivel de impacto cobre este campo com este texto?"""
    return (
        processo_id in NIVEL_COMO_CONSEQUENCIA
        and campo == "consequencia"
        and _sim_nao(_uma_linha(texto)) in NIVEIS_DE_IMPACTO
    )


def _correcao(linha: dict) -> tuple[bool, str]:
    """(ha correcao?, texto novo) a partir de ``texto_corrigido``; ``[VAZIO]`` limpa o campo."""
    bruto = linha["texto_corrigido"].strip()
    if bruto == VAZIO:
        return True, ""
    return bool(bruto), _uma_linha(bruto)


def _coberta_pela_regra(linha: dict) -> bool:
    return _so_nivel(linha["processo_id"], linha["campo"], linha["texto_extraido"])


def _validar(conferencia: list[dict], riscos: dict[tuple[str, str], dict]) -> list[str]:
    problemas: list[str] = []
    vistos: set[tuple[str, str, str]] = set()
    for numero, linha in enumerate(conferencia, start=2):  # linha 1 e o cabecalho
        if linha["processo_id"].startswith(MARCA_METADADO):
            continue
        rotulo = f"linha {numero} ({linha['processo_id']} {linha['risco_id']} {linha['campo']})"
        chave = (linha["processo_id"], linha["risco_id"], linha["campo"])
        risco = riscos.get(chave[:2])
        if linha["campo"] not in CAMPOS:
            problemas.append(f"{rotulo}: campo desconhecido.")
            continue
        if risco is None:
            problemas.append(f"{rotulo}: risco nao existe em {extr.RISCOS_CSV}.")
            continue
        if chave in vistos:
            problemas.append(f"{rotulo}: linha repetida.")
        vistos.add(chave)
        ok = _sim_nao(linha["ok_autor"])
        tem_correcao, corrigido = _correcao(linha)
        pela_regra = _coberta_pela_regra(linha)  # a regra do nivel de impacto limpa este campo
        if ok not in ("sim", "nao"):
            problemas.append(f"{rotulo}: ok_autor vazio ou diferente de sim/nao ('{linha['ok_autor']}').")
        elif ok == "nao" and not tem_correcao and not pela_regra:
            problemas.append(f"{rotulo}: ok_autor=nao sem texto_corrigido (use {VAZIO} para limpar o campo).")
        elif ok == "sim" and tem_correcao:
            problemas.append(f"{rotulo}: ok_autor=sim com texto_corrigido preenchido; qual dos dois vale?")
        elif tem_correcao and not corrigido and linha["campo"] == "risco":
            problemas.append(f"{rotulo}: {VAZIO} nao vale para o enunciado do risco.")
        atual = _uma_linha(risco[linha["campo"]])
        ja_aplicado = (ok == "nao" and tem_correcao and corrigido == atual) or (pela_regra and atual == "")
        if _uma_linha(linha["texto_extraido"]) != atual and not ja_aplicado:
            problemas.append(
                f"{rotulo}: texto_extraido nao confere com {extr.RISCOS_CSV} "
                "(a extracao mudou depois do --gerar, ou o Excel alterou a celula)."
            )
    return problemas


def aplicar(corpus: Path, *, agora: datetime | None = None, saida: Callable[[str], None] = print) -> dict:
    """Aplica a conferencia preenchida; devolve o resumo (tambem impresso)."""
    conferencia = ler_tabela(corpus / CONFERENCIA_CSV, CONFERENCIA_FIELDS)
    linhas_riscos = ler_tabela(corpus / extr.RISCOS_CSV, extr.RISCOS_FIELDS)
    riscos = {(r["processo_id"], r["risco_id"]): r for r in linhas_riscos}

    problemas = _validar(conferencia, riscos)
    if problemas:
        mostrados = "\n".join(f"  - {p}" for p in problemas[:25])
        resto = f"\n  ... e mais {len(problemas) - 25}." if len(problemas) > 25 else ""
        raise ConferenciaError(
            f"{CONFERENCIA_CSV} tem {len(problemas)} linha(s) com problema; nada foi alterado:\n{mostrados}{resto}"
        )

    original = corpus / ORIGINAL_CSV
    if not original.exists():
        shutil.copy2(corpus / extr.RISCOS_CSV, original)
        saida(f"Copia intocada da extracao guardada em {ORIGINAL_CSV}.")

    data_hora = (agora or datetime.now()).isoformat(timespec="seconds")
    linhas = [linha for linha in conferencia if not linha["processo_id"].startswith(MARCA_METADADO)]
    registro: list[dict] = []
    corrigidos: set[tuple[str, str]] = set()

    def registrar(linha: dict, campo: str, antes: str, depois: str) -> None:
        registro.append(
            {
                "processo_id": linha["processo_id"],
                "risco_id": linha["risco_id"],
                "campo": campo,
                "texto_antes": antes,
                "texto_depois": depois,
                "motivo": linha["motivo"],
                "data_hora": data_hora,
            }
        )

    for linha in linhas:
        if _sim_nao(linha["ok_autor"]) != "nao" or _coberta_pela_regra(linha):
            continue
        risco = riscos[(linha["processo_id"], linha["risco_id"])]
        novo = _correcao(linha)[1]
        corrigidos.add((linha["processo_id"], linha["risco_id"]))
        if risco[linha["campo"]] != novo:
            registrar(linha, linha["campo"], risco[linha["campo"]], novo)
            risco[linha["campo"]] = novo
        risco["conferir"] = "corrigido"
    for linha in linhas:
        chave = (linha["processo_id"], linha["risco_id"])
        risco = riscos[chave]
        if _sim_nao(linha["ok_autor"]) == "sim" and chave not in corrigidos and _sim_nao(risco["conferir"]) == "sim":
            registrar(linha, "conferir", risco["conferir"], "conferido")
            risco["conferir"] = "conferido"

    # Regra fixa: nivel de impacto gravado no lugar da consequencia (vale para
    # todos os riscos do processo, estejam ou nao no arquivo de conferencia).
    limpos_pela_regra: Counter = Counter()
    for risco in linhas_riscos:
        if _so_nivel(risco["processo_id"], "consequencia", risco["consequencia"]):
            registrar({**risco, "motivo": MOTIVO_NIVEL}, "consequencia", risco["consequencia"], "")
            risco["consequencia"] = ""
            risco["conferir"] = "corrigido"
            limpos_pela_regra[risco["processo_id"]] += 1

    gravar_tabela(corpus / extr.RISCOS_CSV, extr.RISCOS_FIELDS, linhas_riscos)
    if registro:
        gravar_tabela(corpus / CORRECOES_CSV, CORRECOES_FIELDS, registro, acrescentar=True)

    # Resumo. Campo coberto pela regra do nivel conta na coluna da regra, nao
    # como corrigido nem como confirmado (a regra vale mesmo com ok_autor=sim).
    por_processo: dict[str, Counter] = defaultdict(Counter)
    for linha in linhas:
        if _coberta_pela_regra(linha):
            continue
        por_processo[linha["processo_id"]]["corrigidos" if _sim_nao(linha["ok_autor"]) == "nao" else "confirmados"] += 1
    regra = Counter(
        linha["processo_id"]
        for linha in ler_tabela(corpus / CORRECOES_CSV, CORRECOES_FIELDS)
        if linha["motivo"] == MOTIVO_NIVEL
    ) if (corpus / CORRECOES_CSV).is_file() else Counter()
    for processo, quantidade in regra.items():
        por_processo[processo]["limpos_pela_regra"] = quantidade

    amostra = [linha for linha in linhas if linha["motivo"] == MOTIVO_AMOSTRA]
    tipos: Counter = Counter()
    for linha in amostra:
        if _sim_nao(linha["ok_autor"]) == "nao":
            limpo = _coberta_pela_regra(linha) or _correcao(linha)[1] == ""
            tipos[ERRO_CAMPO_TROCADO if limpo else ERRO_TRANSCRICAO] += 1
    erros_amostra = sum(tipos.values())
    taxa = (lambda n: n / len(amostra) if amostra else None)
    resumo = {
        "por_processo": {p: dict(c) for p, c in sorted(por_processo.items())},
        "corrigidos": sum(c["corrigidos"] for c in por_processo.values()),
        "confirmados": sum(c["confirmados"] for c in por_processo.values()),
        "limpos_pela_regra": sum(regra.values()),
        "alteracoes_gravadas": len(registro),
        "amostra_linhas": len(amostra),
        "amostra_riscos": len({(l["processo_id"], l["risco_id"]) for l in amostra}),
        "amostra_erros": erros_amostra,
        "amostra_erros_transcricao": tipos[ERRO_TRANSCRICAO],
        "amostra_erros_campo_trocado": tipos[ERRO_CAMPO_TROCADO],
        "amostra_taxa_erro": taxa(erros_amostra),
        "amostra_taxa_transcricao": taxa(tipos[ERRO_TRANSCRICAO]),
        "amostra_taxa_campo_trocado": taxa(tipos[ERRO_CAMPO_TROCADO]),
    }

    saida("processo;campos_corrigidos;campos_confirmados;limpos_pela_regra_do_nivel")
    for processo, contagem in resumo["por_processo"].items():
        saida(
            f"{processo};{contagem.get('corrigidos', 0)};{contagem.get('confirmados', 0)};"
            f"{contagem.get('limpos_pela_regra', 0)}"
        )
    saida(f"TOTAL;{resumo['corrigidos']};{resumo['confirmados']};{resumo['limpos_pela_regra']}")
    saida(f"Alteracoes gravadas nesta execucao: {len(registro)} (registro em {CORRECOES_CSV}).")
    if amostra:
        saida(
            f"Amostra aleatoria: {len(amostra)} linhas (campos) de {resumo['amostra_riscos']} riscos; "
            f"erros: {erros_amostra} = {resumo['amostra_taxa_erro']:.1%}"
        )
        saida(
            f"  erros de transcricao (texto reescrito): {tipos[ERRO_TRANSCRICAO]}/{len(amostra)} "
            f"= {resumo['amostra_taxa_transcricao']:.1%}"
        )
        saida(
            f"  erros de campo trocado (campo limpo): {tipos[ERRO_CAMPO_TROCADO]}/{len(amostra)} "
            f"= {resumo['amostra_taxa_campo_trocado']:.1%}"
        )
    return resumo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", required=True, type=Path, help="Pasta do corpus.")
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--gerar", action="store_true", help=f"Grava {CONFERENCIA_CSV}.")
    modo.add_argument("--aplicar", action="store_true", help=f"Aplica o {CONFERENCIA_CSV} preenchido.")
    parser.add_argument("--semente", type=int, default=SEMENTE_PADRAO, help="Semente da amostra (--gerar).")
    parser.add_argument("--sobrescrever", action="store_true", help=f"Permite regravar {CONFERENCIA_CSV}.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus = args.corpus.resolve()
    try:
        if args.gerar:
            gerar(corpus, semente=args.semente, sobrescrever=args.sobrescrever)
        else:
            aplicar(corpus)
    except (ConferenciaError, extr.ExtracaoError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
