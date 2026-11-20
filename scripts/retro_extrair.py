"""Avaliacao retrospectiva, Parte 0: extrai os riscos dos Mapas de Gerenciamento de Riscos em PDF.

Para cada processo com ``incluir=sim`` em ``<corpus>/corpus_casos.csv``, le o
intervalo ``paginas_mapa`` de ``<corpus>/pdfs/<arquivo>``, envia o texto ao
modelo configurado no projeto (``AI_PROVIDER``/``AI_MODEL``, temperatura 0,
prompt versionado em ``prompts/extrair_riscos_v1.txt``) e grava, na pasta do
corpus (nunca no repositorio):

- ``corpus_riscos_reais.csv``: uma linha por risco, com a pagina de origem e
  ``conferir=sim`` quando o enunciado nao foi achado literalmente na pagina;
- ``extracao_log.csv``: modelo, versao do prompt, data e hora, paginas lidas
  e quantidade de riscos por processo.

A extracao e retomavel: processo que ja esta no log e pulado, a nao ser com
``--refazer <processo_id>``. ``corpus_casos.csv`` nunca e regravado.

Checagem literal: o enunciado e procurado na pagina indicada depois de
remover espacos e hifens dos dois lados (cobre quebra de linha, hifenizacao
de fim de linha e espacos a mais ou a menos). Como o texto de tabelas sai do
PDF com as colunas intercaladas, a busca e feita em tres leituras da mesma
pagina: texto corrido, celulas de tabela e faixa vertical (coluna) em que as
palavras do enunciado estao. Nenhuma delas reordena palavras.

Uso:
    python scripts/retro_extrair.py --corpus "<pasta do corpus>"
        [--processos CGU-01 NOVACAP-01] [--refazer CGU-01]

Dependencia so deste script: ``pdfplumber`` (``pip install pdfplumber``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_provider import AIProvider, AIProviderError, build_provider, load_ai_config  # noqa: E402

PROMPT_VERSION = "v1"
PROMPT_PATH = ROOT / "prompts" / "extrair_riscos_v1.txt"

CASOS_CSV = "corpus_casos.csv"
RISCOS_CSV = "corpus_riscos_reais.csv"
LOG_CSV = "extracao_log.csv"
PDF_DIR = "pdfs"

RISCOS_FIELDS = [
    "processo_id",
    "risco_id",
    "fase",
    "risco",
    "causa",
    "consequencia",
    "probabilidade",
    "impacto",
    "pagina",
    "conferir",
]
LOG_FIELDS = [
    "processo_id",
    "provedor",
    "modelo",
    "versao_prompt",
    "data_hora",
    "paginas_lidas",
    "qtd_riscos",
    "qtd_conferir",
]
CASOS_REQUIRED = ["processo_id", "arquivo", "paginas_mapa", "incluir"]

DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT_SECONDS = 600.0
MIN_CHAVE = 8  # enunciado menor que isso casa com qualquer coisa: sempre conferir
MAX_INICIOS_COLUNA = 12  # ocorrencias da primeira palavra do trecho testadas por pagina
MAX_PALAVRAS_COLUNA = 500  # ate onde seguir a leitura depois da primeira palavra
PASSO_COLUNA = 3.0  # pontos; resolucao dos limites de faixa testados
RECUO_COLUNA = 150.0  # quanto a coluna pode comecar a esquerda da primeira palavra
LARGURA_COLUNA = 1000.0  # largura maxima de faixa (texto pode ocupar a pagina toda)

_PAGINAS_RE = re.compile(r"^\s*(\d+)\s*a\s*(\d+)\s*$", re.IGNORECASE)
_HIFENS = "-‐‑­"
_ASPAS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
_DESCARTAR_RE = re.compile(rf"[\s{re.escape(_HIFENS)}]+")


class ExtracaoError(Exception):
    """Falha que impede a extracao (corpus invalido, PDF ou resposta do modelo)."""


# --------------------------------------------------------------------------
# corpus_casos.csv (editado no Excel: leitura tolerante, nunca regravado)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Caso:
    processo_id: str
    arquivo: str
    pagina_inicial: int
    pagina_final: int
    observacao: str


def _sem_acento(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto or "")
    return "".join(ch for ch in decomposto if not unicodedata.combining(ch))


def _decodificar(dados: bytes) -> str:
    """UTF-8 (com ou sem BOM) ou, se nao for UTF-8 valido, Windows-1252."""
    try:
        return dados.decode("utf-8-sig")
    except UnicodeDecodeError:
        return dados.decode("cp1252")


def _detectar_separador(cabecalho: str) -> str:
    return "\t" if cabecalho.count("\t") > cabecalho.count(";") else ";"


def _ler_linhas_csv(caminho: Path) -> list[dict[str, str]]:
    texto = _decodificar(caminho.read_bytes())
    linhas = texto.splitlines()
    if not linhas:
        return []
    leitor = csv.reader(linhas, delimiter=_detectar_separador(linhas[0]))
    cabecalho = [coluna.strip().lower() for coluna in next(leitor)]
    registros = []
    for valores in leitor:
        if not any(valor.strip() for valor in valores):
            continue
        valores = valores + [""] * (len(cabecalho) - len(valores))
        registros.append({coluna: valor.strip() for coluna, valor in zip(cabecalho, valores)})
    return registros


def deve_incluir(valor: str) -> bool:
    """``sim`` sem diferenciar maiusculas nem acento; qualquer outra coisa e nao."""
    return _sem_acento(valor).strip().casefold() == "sim"


def interpretar_paginas(valor: str, processo_id: str) -> tuple[int, int]:
    """``29 a 40`` -> (29, 40). Outro formato (ex.: data criada pelo Excel) e erro."""
    casado = _PAGINAS_RE.match(valor or "")
    if not casado:
        raise ExtracaoError(
            f"{processo_id}: paginas_mapa='{valor}' nao esta no formato '29 a 40' "
            "(o Excel pode ter convertido o campo em data). Corrija o corpus_casos.csv."
        )
    inicial, final = int(casado.group(1)), int(casado.group(2))
    if inicial < 1 or final < inicial:
        raise ExtracaoError(f"{processo_id}: paginas_mapa='{valor}' tem intervalo invalido.")
    return inicial, final


def ler_casos(caminho: Path) -> list[Caso]:
    """Devolve so os processos com ``incluir=sim``, ja com ``paginas_mapa`` validado."""
    if not caminho.is_file():
        raise ExtracaoError(f"Arquivo nao encontrado: {caminho}")
    registros = _ler_linhas_csv(caminho)
    if not registros:
        raise ExtracaoError(f"{caminho.name} esta vazio.")
    faltando = [coluna for coluna in CASOS_REQUIRED if coluna not in registros[0]]
    if faltando:
        raise ExtracaoError(f"{caminho.name} sem as colunas: {', '.join(faltando)}")

    casos: list[Caso] = []
    vistos: set[str] = set()
    for registro in registros:
        if not deve_incluir(registro["incluir"]):
            continue
        processo_id = registro["processo_id"]
        if not processo_id:
            raise ExtracaoError(f"{caminho.name}: linha com incluir=sim sem processo_id.")
        if processo_id in vistos:
            raise ExtracaoError(f"{caminho.name}: processo_id repetido: {processo_id}")
        vistos.add(processo_id)
        inicial, final = interpretar_paginas(registro["paginas_mapa"], processo_id)
        casos.append(
            Caso(
                processo_id=processo_id,
                arquivo=registro["arquivo"] or f"{processo_id}.pdf",
                pagina_inicial=inicial,
                pagina_final=final,
                observacao=registro.get("observacao", ""),
            )
        )
    return casos


# --------------------------------------------------------------------------
# Texto do PDF, pagina a pagina
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Palavra:
    texto: str
    x0: float
    x1: float


@dataclass
class Pagina:
    numero: int
    texto: str
    layout: str = ""  # texto com as colunas afastadas por espacos, como na pagina
    tabelas: list[list[list[str]]] = field(default_factory=list)
    palavras: list[Palavra] = field(default_factory=list)  # na ordem de leitura


def extrair_paginas(pdf_path: Path, inicial: int, final: int) -> list[Pagina]:
    """Le so as paginas ``inicial``..``final`` (inclusive, a partir de 1) do PDF."""
    import pdfplumber  # import tardio: so exigido em uso real

    paginas: list[Pagina] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        if final > total:
            raise ExtracaoError(
                f"{pdf_path.name}: paginas_mapa vai ate a pagina {final}, mas o PDF tem {total}."
            )
        for numero in range(inicial, final + 1):
            page = pdf.pages[numero - 1]
            paginas.append(
                Pagina(
                    numero=numero,
                    texto=page.extract_text() or "",
                    layout=page.extract_text(layout=True) or "",
                    tabelas=[
                        [[celula or "" for celula in linha] for linha in tabela]
                        for tabela in page.extract_tables()
                    ],
                    palavras=[
                        Palavra(texto=w["text"], x0=float(w["x0"]), x1=float(w["x1"]))
                        for w in page.extract_words()
                    ],
                )
            )


    if not any(pagina.texto.strip() for pagina in paginas):
        raise ExtracaoError(f"{pdf_path.name}: paginas {inicial} a {final} sem camada de texto.")
    return paginas


def _linha_tabela(linha: list[str]) -> str:
    return " | ".join(" ".join(celula.split()) for celula in linha)


def montar_documento(paginas: Iterable[Pagina]) -> str:
    """Texto enviado ao modelo: cada pagina em ate duas leituras (ver o prompt)."""
    partes: list[str] = []
    for pagina in paginas:
        partes.append(f"===== PÁGINA {pagina.numero} =====")
        partes.append("[TEXTO]")
        linhas = (pagina.layout or pagina.texto).splitlines()
        partes.extend(linha.rstrip() for linha in linhas if linha.strip())
        tabelas = [t for t in pagina.tabelas if any(sum(1 for c in linha if c.strip()) > 1 for linha in t)]
        if tabelas:
            partes.append("[TABELAS]")
            for indice, tabela in enumerate(tabelas, start=1):
                partes.append(f"Tabela {indice}:")
                partes.extend(_linha_tabela(linha) for linha in tabela)
        partes.append("")
    return "\n".join(partes)


def montar_prompt(caso: Caso, paginas: list[Pagina], template: str | None = None) -> str:
    template = template if template is not None else PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("<<PROCESSO_ID>>", caso.processo_id)
        .replace("<<PAGINAS>>", f"{caso.pagina_inicial} a {caso.pagina_final}")
        .replace("<<OBSERVACAO>>", caso.observacao or "(nenhuma)")
        .replace("<<DOCUMENTO>>", montar_documento(paginas))
    )


# --------------------------------------------------------------------------
# Resposta do modelo: JSON bem formado e campos esperados
# --------------------------------------------------------------------------


def _uma_linha(valor: object) -> str:
    if valor is None:
        return ""
    return " ".join(str(valor).split())


def _campo(valor: object) -> str:
    """Texto em uma linha; lista vira itens unidos por ' | '."""
    if isinstance(valor, (list, tuple)):
        return " | ".join(item for item in (_uma_linha(v) for v in valor) if item)
    return _uma_linha(valor)


def _pagina(valor: object) -> int | None:
    if isinstance(valor, bool):
        return None
    if isinstance(valor, int):
        return valor
    texto = _uma_linha(valor)
    return int(texto) if texto.isdigit() else None


def interpretar_resposta(bruto: str) -> list[dict]:
    """Valida o JSON do modelo e devolve os riscos com os campos normalizados.

    Lanca ``ExtracaoError`` se a resposta nao for um JSON bem formado no
    formato pedido (inclusive resposta cortada) ou se algum item vier sem
    enunciado de risco.
    """
    texto = (bruto or "").strip()
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim <= inicio:
        raise ExtracaoError("Resposta do modelo nao contem um objeto JSON.")
    try:
        dados = json.loads(texto[inicio : fim + 1])
    except json.JSONDecodeError as exc:
        raise ExtracaoError(f"Resposta do modelo nao e um JSON valido (cortada?): {exc}") from exc
    if not isinstance(dados, dict) or not isinstance(dados.get("riscos"), list):
        raise ExtracaoError('Resposta do modelo sem a lista "riscos".')

    riscos: list[dict] = []
    for ordem, item in enumerate(dados["riscos"], start=1):
        if not isinstance(item, dict):
            raise ExtracaoError(f"Item {ordem} da resposta nao e um objeto JSON.")
        enunciado = _campo(item.get("risco"))
        if not enunciado:
            raise ExtracaoError(f"Item {ordem} da resposta veio sem o campo 'risco'.")
        riscos.append(
            {
                "risco_id": _campo(item.get("risco_id")) or str(ordem),
                "fase": _campo(item.get("fase")),
                "risco": enunciado,
                "causa": _campo(item.get("causa")),
                "consequencia": _campo(item.get("consequencia")),
                "probabilidade": _campo(item.get("probabilidade")),
                "impacto": _campo(item.get("impacto")),
                "pagina": _pagina(item.get("pagina")),
            }
        )
    return riscos


# --------------------------------------------------------------------------
# Checagem literal do enunciado na pagina indicada
# --------------------------------------------------------------------------


def chave(texto: str) -> str:
    """Forma de comparacao: sem espacos e sem hifens (quebra e hifenizacao de linha)."""
    normalizado = unicodedata.normalize("NFKC", texto or "").translate(_ASPAS)
    return _DESCARTAR_RE.sub("", normalizado)


def _achar_em_coluna(trecho: str, palavras: list[Palavra]) -> bool:
    """Acha o trecho, sem nada no meio, dentro de uma faixa vertical da pagina.

    Em tabela sem bordas, o texto de uma celula sai intercalado com o das
    vizinhas. Para cada palavra da pagina que pode iniciar o trecho, testa as
    faixas horizontais [esquerda, direita] em volta dela (os limites vem das
    bordas das palavras da pagina): lendo, na ordem da pagina, so as palavras
    inteiramente dentro da faixa, o trecho precisa aparecer inteiro e
    contiguo. Uma palavra a mais ou fora de ordem dentro da faixa e nao casa.
    """
    alvo = chave(trecho)
    chaves = [chave(p.texto) for p in palavras]

    def casa_na_faixa(inicio: int, esquerda: float, direita: float) -> bool:
        consumido = 0
        for indice in range(inicio, min(len(palavras), inicio + MAX_PALAVRAS_COLUNA)):
            atual = chaves[indice]
            if not atual or palavras[indice].x0 < esquerda or palavras[indice].x1 > direita:
                continue
            if alvo.startswith(atual, consumido):
                consumido += len(atual)
                if consumido == len(alvo):
                    return True
                continue
            # ultima palavra do trecho sem a pontuacao que a segue na pagina
            return atual.startswith(alvo[consumido:])
        return False

    def grade(valores: Iterable[float]) -> list[float]:
        return sorted({PASSO_COLUNA * round(valor / PASSO_COLUNA) for valor in valores})

    inicios = [i for i, atual in enumerate(chaves) if atual and alvo.startswith(atual)]
    for inicio in inicios[:MAX_INICIOS_COLUNA]:
        primeira = palavras[inicio]
        esquerdas = grade(
            p.x0 for p in palavras if primeira.x0 - RECUO_COLUNA <= p.x0 <= primeira.x0
        )
        direitas = grade(
            p.x1 for p in palavras if primeira.x1 <= p.x1 <= primeira.x0 + LARGURA_COLUNA
        )
        for esquerda in reversed(esquerdas):
            for direita in direitas:
                if casa_na_faixa(inicio, esquerda - PASSO_COLUNA, direita + PASSO_COLUNA):
                    return True
    return False


def localizar_trecho(trecho: str, pagina: Pagina, proxima: Pagina | None = None) -> str | None:
    """Diz em qual leitura da pagina o trecho foi achado literalmente, ou ``None``."""
    alvo = chave(trecho)
    if len(alvo) < MIN_CHAVE:
        return None
    if alvo in chave(pagina.texto):
        return "texto"
    if any(alvo in chave(celula) for tabela in pagina.tabelas for linha in tabela for celula in linha):
        return "tabela"
    if _achar_em_coluna(trecho, pagina.palavras):
        return "coluna"
    if (
        proxima is not None
        and alvo in chave(pagina.texto + " " + proxima.texto)
        and alvo not in chave(proxima.texto)  # inteiro na pagina seguinte = pagina errada
    ):
        return "texto+proxima"  # enunciado atravessa a quebra de pagina
    return None


def marcar_conferir(riscos: list[dict], paginas: list[Pagina]) -> None:
    """Preenche ``conferir`` (sim/nao) em cada risco, pela checagem literal."""
    por_numero = {pagina.numero: pagina for pagina in paginas}
    for risco in riscos:
        pagina = por_numero.get(risco["pagina"]) if risco["pagina"] is not None else None
        achado = (
            localizar_trecho(risco["risco"], pagina, por_numero.get(pagina.numero + 1))
            if pagina is not None
            else None
        )
        risco["conferir"] = "nao" if achado else "sim"


# --------------------------------------------------------------------------
# Arquivos de saida (pasta do corpus) e retomada
# --------------------------------------------------------------------------


def _ler_saida(caminho: Path) -> list[dict[str, str]]:
    if not caminho.is_file():
        return []
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


def _gravar(caminho: Path, campos: list[str], linhas: list[dict], *, acrescentar: bool) -> None:
    """CSV com ``;`` e UTF-8 com BOM (o BOM so entra quando o arquivo e criado)."""
    novo = not (acrescentar and caminho.is_file())
    try:
        with caminho.open("w" if novo else "a", encoding="utf-8-sig" if novo else "utf-8", newline="") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";", extrasaction="ignore")
            if novo:
                escritor.writeheader()
            escritor.writerows(linhas)
    except PermissionError as exc:
        raise ExtracaoError(f"Nao foi possivel gravar {caminho.name} (aberto no Excel?): {exc}") from exc


def processos_extraidos(corpus: Path) -> set[str]:
    return {linha["processo_id"] for linha in _ler_saida(corpus / LOG_CSV)}


def _gravar_processo(corpus: Path, caso: Caso, riscos: list[dict], log: dict, *, substituir: bool) -> None:
    linhas = [{**risco, "processo_id": caso.processo_id, "pagina": risco["pagina"] or ""} for risco in riscos]
    riscos_csv = corpus / RISCOS_CSV
    if substituir and riscos_csv.is_file():
        mantidas = [l for l in _ler_saida(riscos_csv) if l["processo_id"] != caso.processo_id]
        _gravar(riscos_csv, RISCOS_FIELDS, mantidas + linhas, acrescentar=False)
    else:
        _gravar(riscos_csv, RISCOS_FIELDS, linhas, acrescentar=True)
    _gravar(corpus / LOG_CSV, LOG_FIELDS, [log], acrescentar=True)


# --------------------------------------------------------------------------
# Execucao
# --------------------------------------------------------------------------

Leitor = Callable[[Path, int, int], list[Pagina]]


def extrair_processo(
    caso: Caso, pdf_path: Path, provider: AIProvider, leitor: Leitor = extrair_paginas
) -> tuple[list[dict], list[Pagina]]:
    paginas = leitor(pdf_path, caso.pagina_inicial, caso.pagina_final)
    try:
        bruto = provider.generate(montar_prompt(caso, paginas))
    except AIProviderError as exc:
        raise ExtracaoError(f"Falha na chamada ao modelo: {exc}") from exc
    riscos = interpretar_resposta(bruto)
    marcar_conferir(riscos, paginas)
    return riscos, paginas


def executar(
    corpus: Path,
    provider: AIProvider,
    *,
    provedor: str,
    modelo: str,
    processos: Iterable[str] | None = None,
    refazer: Iterable[str] = (),
    leitor: Leitor = extrair_paginas,
    saida: Callable[[str], None] = print,
) -> int:
    """Extrai os processos pendentes; devolve quantos falharam."""
    casos = ler_casos(corpus / CASOS_CSV)
    conhecidos = {caso.processo_id for caso in casos}
    refazer = set(refazer)
    pedidos = set(processos) if processos else None
    desconhecidos = sorted((refazer | (pedidos or set())) - conhecidos)
    if desconhecidos:
        raise ExtracaoError(f"Processo fora do corpus ou com incluir diferente de sim: {', '.join(desconhecidos)}")
    if pedidos is not None:
        casos = [caso for caso in casos if caso.processo_id in pedidos | refazer]

    feitos = processos_extraidos(corpus)
    pendentes = [caso for caso in casos if caso.processo_id not in feitos or caso.processo_id in refazer]
    for caso in pendentes:
        if not (corpus / PDF_DIR / caso.arquivo).is_file():
            raise ExtracaoError(f"{caso.processo_id}: PDF nao encontrado em {PDF_DIR}/{caso.arquivo}")
    saida(f"{len(casos)} processo(s) selecionado(s); {len(casos) - len(pendentes)} ja extraido(s), {len(pendentes)} a extrair.")

    falhas = 0
    for caso in pendentes:
        try:
            riscos, _ = extrair_processo(caso, corpus / PDF_DIR / caso.arquivo, provider, leitor)
        except ExtracaoError as exc:
            falhas += 1
            saida(f"[FALHA] {caso.processo_id}: {exc}")
            continue
        conferir = [risco["risco_id"] for risco in riscos if risco["conferir"] == "sim"]
        log = {
            "processo_id": caso.processo_id,
            "provedor": provedor,
            "modelo": modelo,
            "versao_prompt": PROMPT_VERSION,
            "data_hora": datetime.now().isoformat(timespec="seconds"),
            "paginas_lidas": f"{caso.pagina_inicial} a {caso.pagina_final}",
            "qtd_riscos": len(riscos),
            "qtd_conferir": len(conferir),
        }
        _gravar_processo(corpus, caso, riscos, log, substituir=caso.processo_id in feitos)
        aviso = f" (conferir: {', '.join(conferir)})" if conferir else ""
        vazio = " ATENCAO: nenhum risco extraido." if not riscos else ""
        saida(f"[ok] {caso.processo_id}: {len(riscos)} risco(s), {len(conferir)} com conferir=sim{aviso}{vazio}")
    return falhas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", required=True, type=Path, help="Pasta do corpus (com corpus_casos.csv e pdfs/).")
    parser.add_argument("--processos", nargs="+", metavar="ID", help="Extrair so estes processos.")
    parser.add_argument("--refazer", nargs="+", default=[], metavar="ID", help="Extrair de novo estes processos.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"Pasta do corpus nao encontrada: {corpus}", file=sys.stderr)
        return 2
    os.chdir(ROOT)  # .streamlit/secrets.toml e lido a partir da raiz do projeto
    try:
        config = load_ai_config()
        provider = build_provider(config, timeout=args.timeout)
        if hasattr(provider, "max_tokens"):
            provider.max_tokens = args.max_tokens  # o padrao do app (2048) corta a lista
        print(f"Modelo: {config.provider}/{config.model} | prompt {PROMPT_VERSION} | corpus: {corpus}")
        falhas = executar(
            corpus,
            provider,
            provedor=config.provider,
            modelo=config.model,
            processos=args.processos,
            refazer=args.refazer,
        )
    except (ExtracaoError, AIProviderError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
