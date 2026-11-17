"""Avaliacao retrospectiva, Parte 2: julgamento por um modelo juiz.

Dois modos, ambos retomaveis (pulam o que ja foi julgado) e com ``--limite N``
(so os N primeiros processos incluidos):

- cobertura (padrao): para cada risco real, o juiz recebe a lista de candidatos
  do mesmo processo, SEM a origem (codigos neutros no lugar do ``candidato_id``)
  e em ordem embaralhada, e indica quais descrevem o mesmo evento de risco.
  Grava ``julgamento.csv``; a coluna ``cobertura`` (catalogo|ia|ambos|nenhum) e
  derivada aqui, a partir da origem dos candidatos marcados.
- ``--riscos-ia``: uma chamada por processo; o juiz diz, para cada risco de IA,
  se o texto nomeia o objeto da contratacao (e com que termo), se o risco e
  especifico daquele objeto e a que grupo de riscos repetidos ele pertence.
  Grava ``riscos_ia.csv``; o termo que nao aparece literalmente no texto do
  risco deixa a linha marcada na coluna ``conferir``.

O juiz vem de ``JUDGE_PROVIDER``/``JUDGE_MODEL`` (variavel de ambiente ou
``.streamlit/secrets.toml``) e tem de ser diferente do modelo gerador. Tudo e
gravado em ``<corpus>/retrospectiva/``; nada no repositorio.

Uso:
    python scripts/retro_julgar.py --corpus "<pasta do corpus>" [--limite N] [--dry-run]
    python scripts/retro_julgar.py --corpus "<pasta do corpus>" --riscos-ia [--limite N] [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
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
from src import ai_provider  # noqa: E402
from src.ai_provider import AIProvider, AIProviderError  # noqa: E402
from src.catalog_mix import ORIGIN_AI, ORIGIN_CATALOG  # noqa: E402

PROMPT_COBERTURA = ROOT / "prompts" / "juiz_cobertura_v1.txt"
VERSAO_COBERTURA = "v1"
PROMPT_RISCOS_IA = ROOT / "prompts" / "juiz_riscos_ia_v2.txt"  # o v1 fica no repositorio como registro do piloto
VERSAO_RISCOS_IA = "v2"

JULGAMENTO_CSV = "julgamento.csv"
RISCOS_IA_CSV = "riscos_ia.csv"
JULGAMENTO_FIELDS = [
    "processo_id",
    "risco_id",
    "candidatos_cobertos",
    "cobertura",
    "justificativa",
    "modelo_juiz",
    "versao_prompt_juiz",
    "semente",
]
RISCOS_IA_FIELDS = [
    "processo_id",
    "candidato_id",
    "categoria",
    "nomeia_objeto",
    "termo",
    "especifico_do_objeto",
    "grupo",
    "modelo_juiz",
    "versao_prompt_juiz",
    "conferir",  # "nao", ou o motivo para o autor conferir a linha
]

SEMENTE_PADRAO = 20261117
TIMEOUT_PADRAO = 120.0
MAX_TOKENS_JUIZ = 8192
NOVAS_TENTATIVAS = 1  # do provedor (rede/timeout) e, aqui, de resposta invalida
FALHAS_SEGUIDAS_MAX = 3  # o provedor esta recusando: nao adianta insistir
SEPARADOR = " | "
NAO_INFORMADA = "(não informada)"


class JulgamentoError(Exception):
    """Falha que impede preparar ou gravar o julgamento."""


class RespostaInvalida(Exception):
    """A resposta do juiz nao tem o formato pedido."""


# --------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------


def ler_candidatos(saida_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Candidatos por processo, na ordem de ``candidatos.csv``."""
    linhas = conf.ler_tabela(saida_dir / gerar.CANDIDATOS_CSV, gerar.CANDIDATOS_FIELDS)
    por_processo: dict[str, list[dict[str, str]]] = {}
    for linha in linhas:
        if linha["origem"] not in (ORIGIN_CATALOG, ORIGIN_AI):
            raise JulgamentoError(f"{linha['candidato_id']}: origem desconhecida '{linha['origem']}'.")
        por_processo.setdefault(linha["processo_id"], []).append(linha)
    return por_processo


def ler_riscos_reais(corpus: Path) -> dict[str, list[dict[str, str]]]:
    """Riscos reais (com enunciado) por processo, na ordem do arquivo."""
    linhas = conf.ler_tabela(corpus / gerar.RISCOS_CSV, ["processo_id", "risco_id", "risco", "causa", "consequencia"])
    por_processo: dict[str, list[dict[str, str]]] = {}
    vistos: set[tuple[str, str]] = set()
    for linha in linhas:
        if not linha["risco"]:
            continue
        chave = (linha["processo_id"], linha["risco_id"])
        if chave in vistos:
            raise JulgamentoError(f"{gerar.RISCOS_CSV}: risco repetido: {chave[0]} / {chave[1]}")
        vistos.add(chave)
        por_processo.setdefault(linha["processo_id"], []).append(linha)
    return por_processo


def modelos_geradores(saida_dir: Path) -> set[str]:
    """Modelos que geraram os riscos de IA, segundo ``geracao_log.csv``."""
    caminho = saida_dir / gerar.LOG_CSV
    if not caminho.is_file():
        return set()
    return {
        linha["modelo"] for linha in conf.ler_tabela(caminho, ["modelo", "situacao"]) if linha["situacao"] == "gerado"
    }


def _ja_gravadas(caminho: Path, campos: list[str], esperado: dict[str, str]) -> list[dict[str, str]]:
    """Linhas ja gravadas; recusa misturar modelo, prompt ou semente diferentes."""
    if not caminho.is_file():
        return []
    linhas = conf.ler_tabela(caminho, campos)
    for linha in linhas:
        diferentes = [f"{campo}={linha[campo]}" for campo, valor in esperado.items() if linha[campo] != valor]
        if diferentes:
            raise JulgamentoError(
                f"{caminho.name} ja tem linhas com outra configuracao ({', '.join(diferentes)}; agora: "
                f"{', '.join(f'{c}={v}' for c, v in esperado.items())}). Renomeie o arquivo para julgar de novo."
            )
    return linhas


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def embaralhar(candidatos: list[dict[str, str]], semente: int, processo_id: str, risco_id: str) -> list[tuple[str, dict]]:
    """Candidatos em ordem embaralhada, com codigos neutros (``C01``...).

    A ordem depende so de ``(semente, processo_id, risco_id)``: a retomada e uma
    nova execucao com a mesma semente apresentam a mesma lista ao juiz.
    """
    ordem = list(candidatos)
    random.Random(f"{semente}|{processo_id}|{risco_id}").shuffle(ordem)
    return [(f"C{posicao:02d}", candidato) for posicao, candidato in enumerate(ordem, start=1)]


def _ou_nao_informada(texto: str) -> str:
    return texto or NAO_INFORMADA


def prompt_cobertura(risco: dict[str, str], codificados: list[tuple[str, dict]]) -> str:
    blocos = [
        f"[{codigo}] Título: {c['titulo']}\n"
        f"      Causa: {_ou_nao_informada(c['causa'])}\n"
        f"      Consequência: {_ou_nao_informada(c['consequencia'])}"
        for codigo, c in codificados
    ]
    return PROMPT_COBERTURA.read_text(encoding="utf-8").format(
        risco=risco["risco"],
        causa=_ou_nao_informada(risco["causa"]),
        consequencia=_ou_nao_informada(risco["consequencia"]),
        candidatos="\n".join(blocos),
    )


def prompt_riscos_ia(processo: gerar.Processo, codificados: list[tuple[str, dict]]) -> str:
    blocos = [
        f"[{codigo}] Categoria: {c['categoria']}\n"
        f"      Título: {c['titulo']}\n"
        f"      Causa: {_ou_nao_informada(c['causa'])}\n"
        f"      Consequência: {_ou_nao_informada(c['consequencia'])}"
        for codigo, c in codificados
    ]
    return PROMPT_RISCOS_IA.read_text(encoding="utf-8").format(
        objeto=processo.objeto, tipo_contratacao=processo.tipo_contratacao, riscos="\n".join(blocos)
    )


# --------------------------------------------------------------------------
# Respostas do juiz
# --------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extrair_json(texto: str) -> dict:
    achado = _JSON_RE.search(texto or "")
    if not achado:
        raise RespostaInvalida("resposta sem objeto JSON")
    try:
        dados = json.loads(achado.group(0), strict=False)  # tolera quebra de linha dentro de texto
    except json.JSONDecodeError as exc:
        raise RespostaInvalida(f"JSON invalido: {exc.msg}") from exc
    if not isinstance(dados, dict):
        raise RespostaInvalida("o JSON nao e um objeto")
    return dados


def _codigo(valor: object) -> str:
    return str(valor).strip().strip("[]").upper()


def interpretar_cobertura(texto: str, codigos: list[str]) -> tuple[list[str], str]:
    """``(codigos cobertos, na ordem apresentada; justificativa)``."""
    dados = _extrair_json(texto)
    cobertos = dados.get("cobertos")
    if not isinstance(cobertos, list):
        raise RespostaInvalida("'cobertos' ausente ou nao e lista")
    marcados = {_codigo(item) for item in cobertos}
    desconhecidos = sorted(marcados - set(codigos))
    if desconhecidos:
        raise RespostaInvalida(f"codigo(s) fora da lista: {', '.join(desconhecidos)}")
    justificativa = dados.get("justificativa")
    if not isinstance(justificativa, str) or not justificativa.strip():
        raise RespostaInvalida("'justificativa' ausente")
    return [codigo for codigo in codigos if codigo in marcados], " ".join(justificativa.split())


_CODIGO_RE = re.compile(r"\bC\d{2,}\b")


def trocar_codigos(texto: str, por_codigo: dict[str, dict]) -> str:
    """Troca os codigos neutros (``C01``...) citados pelo juiz pelos ``candidato_id`` reais."""
    return _CODIGO_RE.sub(
        lambda achado: por_codigo[achado.group(0)]["candidato_id"] if achado.group(0) in por_codigo else achado.group(0),
        texto,
    )


def derivar_cobertura(origens: list[str]) -> str:
    """``catalogo``, ``ia``, ``ambos`` ou ``nenhum``, pela origem dos candidatos marcados."""
    tem_catalogo = ORIGIN_CATALOG in origens
    tem_ia = ORIGIN_AI in origens
    if tem_catalogo and tem_ia:
        return "ambos"
    if tem_catalogo:
        return "catalogo"
    return "ia" if tem_ia else "nenhum"


def _sim_ou_nao(item: dict, campo: str, codigo: str) -> str:
    valor = conf._sim_nao(str(item.get(campo, "")))
    if valor not in ("sim", "nao"):
        raise RespostaInvalida(f"{codigo}: {campo} invalido: {item.get(campo)!r}")
    return valor


_ASPAS = "\"'\u201c\u201d\u2018\u2019 "


def _termos(valor: object, codigo: str) -> list[str]:
    """Trechos citados pelo juiz (texto com `` | `` ou lista), sem aspas em volta."""
    if valor is None:
        return []
    if isinstance(valor, str):
        valor = valor.split("|")
    if not isinstance(valor, list) or not all(isinstance(parte, str) for parte in valor):
        raise RespostaInvalida(f"{codigo}: termo invalido: {valor!r}")
    return [termo for termo in (" ".join(parte.split()).strip(_ASPAS) for parte in valor) if termo]


def interpretar_riscos_ia(texto: str, codigos: list[str]) -> dict[str, dict]:
    """``{codigo: {nomeia_objeto, termos, especifico_do_objeto, grupo}}``, um item por risco."""
    itens = _extrair_json(texto).get("riscos")
    if not isinstance(itens, list):
        raise RespostaInvalida("'riscos' ausente ou nao e lista")
    resultado: dict[str, dict] = {}
    for item in itens:
        if not isinstance(item, dict):
            raise RespostaInvalida("item de 'riscos' nao e objeto")
        codigo = _codigo(item.get("id", ""))
        if codigo not in codigos:
            raise RespostaInvalida(f"id fora da lista: {codigo}")
        if codigo in resultado:
            raise RespostaInvalida(f"id repetido: {codigo}")
        grupo = item.get("grupo")
        if isinstance(grupo, bool) or not (isinstance(grupo, int) or (isinstance(grupo, str) and grupo.strip().isdigit())):
            raise RespostaInvalida(f"{codigo}: grupo invalido: {grupo!r}")
        resultado[codigo] = {
            "nomeia_objeto": _sim_ou_nao(item, "nomeia_objeto", codigo),
            "termos": _termos(item.get("termo"), codigo),
            "especifico_do_objeto": _sim_ou_nao(item, "especifico_do_objeto", codigo),
            "grupo": int(grupo),
        }
    faltando = [codigo for codigo in codigos if codigo not in resultado]
    if faltando:
        raise RespostaInvalida(f"risco(s) sem resposta: {', '.join(faltando)}")
    return resultado


def conferir_termo(candidato: dict[str, str], nomeia_objeto: str, termos: list[str]) -> str:
    """``nao`` se a resposta e coerente com o texto do risco; senao, o motivo para conferir.

    Todo termo tem de existir literalmente (sem diferenciar maiusculas nem
    espacos) no titulo, na causa ou na consequencia do risco.
    """
    campos = [" ".join(candidato[campo].split()).casefold() for campo in ("titulo", "causa", "consequencia")]
    ausentes = [termo for termo in termos if not any(termo.casefold() in campo for campo in campos)]
    motivos = []
    if ausentes:
        motivos.append("termo nao encontrado no texto: " + SEPARADOR.join(ausentes))
    if nomeia_objeto == "sim" and not termos:
        motivos.append("nomeia_objeto=sim sem termo")
    if nomeia_objeto == "nao" and termos:
        motivos.append("termo preenchido com nomeia_objeto=nao")
    return "; ".join(motivos) or "nao"


def _perguntar(provider: AIProvider, prompt: str, interpretar: Callable[[str], object]):
    """Chama o juiz; uma nova tentativa se a resposta vier fora do formato."""
    ultimo: Exception | None = None
    for _ in range(NOVAS_TENTATIVAS + 1):
        try:
            return interpretar(provider.generate(prompt))
        except RespostaInvalida as exc:
            ultimo = exc
    raise ultimo  # type: ignore[misc]


# --------------------------------------------------------------------------
# Execucao
# --------------------------------------------------------------------------


def _preparar(corpus: Path, modelo: str, limite: int | None):
    corpus = corpus.resolve()
    if corpus == ROOT or ROOT in corpus.parents:
        raise JulgamentoError("A pasta do corpus nao pode ficar dentro do repositorio.")
    saida_dir = corpus / gerar.SAIDA_DIR
    try:
        todos = gerar.ler_processos(corpus)
    except gerar.GeracaoError as exc:
        raise JulgamentoError(str(exc)) from exc
    if modelo in modelos_geradores(saida_dir):
        raise JulgamentoError(f"O modelo juiz ({modelo}) e o mesmo que gerou os riscos de IA; escolha outro.")
    candidatos = ler_candidatos(saida_dir)
    processos = todos[:limite] if limite else todos
    sem_candidatos = [p.processo_id for p in processos if p.processo_id not in candidatos]
    if sem_candidatos:
        raise JulgamentoError(f"Processo(s) sem candidatos em {gerar.CANDIDATOS_CSV}: {', '.join(sem_candidatos)}")
    return corpus, saida_dir, todos, processos, candidatos


def julgar_cobertura(
    corpus: Path,
    provider: AIProvider | None,
    *,
    modelo: str,
    semente: int = SEMENTE_PADRAO,
    limite: int | None = None,
    dry_run: bool = False,
    saida: Callable[[str], None] = print,
) -> int:
    """Julga os riscos reais pendentes; devolve quantos ficaram sem julgamento."""
    corpus, saida_dir, todos, processos, candidatos = _preparar(corpus, modelo, limite)
    riscos = ler_riscos_reais(corpus)
    caminho = saida_dir / JULGAMENTO_CSV
    esperado = {"modelo_juiz": modelo, "versao_prompt_juiz": VERSAO_COBERTURA, "semente": str(semente)}
    feitos = {(l["processo_id"], l["risco_id"]) for l in _ja_gravadas(caminho, JULGAMENTO_FIELDS, esperado)}
    pendentes = [
        (processo, risco)
        for processo in processos
        for risco in riscos.get(processo.processo_id, [])
        if (processo.processo_id, risco["risco_id"]) not in feitos
    ]
    total = sum(len(riscos.get(p.processo_id, [])) for p in processos)
    saida(
        f"{len(todos)} processo(s) incluido(s); {len(processos)} nesta execucao; {total} risco(s) reais; "
        f"{total - len(pendentes)} ja julgado(s); {len(pendentes)} a julgar (1 chamada cada)."
    )
    saida(f"Juiz {modelo} | prompt {VERSAO_COBERTURA} | semente {semente} | saida: {caminho}")
    if dry_run:
        saida("[dry-run] nada gravado.")
        return 0
    if provider is None:
        raise JulgamentoError("Modelo juiz nao configurado.")

    falharam = seguidas = 0
    for posicao, (processo, risco) in enumerate(pendentes, start=1):
        pid, rid = processo.processo_id, risco["risco_id"]
        codificados = embaralhar(candidatos[pid], semente, pid, rid)
        por_codigo = dict(codificados)
        codigos = list(por_codigo)
        try:
            cobertos, justificativa = _perguntar(
                provider, prompt_cobertura(risco, codificados), lambda texto: interpretar_cobertura(texto, codigos)
            )
        except (AIProviderError, RespostaInvalida) as exc:
            falharam += 1
            seguidas += 1
            saida(f"[FALHA] {pid} / {rid}: {gerar._resumir_erro(str(exc))}")
            if seguidas >= FALHAS_SEGUIDAS_MAX:
                restantes = len(pendentes) - posicao
                falharam += restantes
                saida(f"{seguidas} falhas seguidas; execucao interrompida ({restantes} risco(s) nem foram tentados).")
                break
            continue
        seguidas = 0
        marcados = [por_codigo[codigo] for codigo in cobertos]
        linha = {
            "processo_id": pid,
            "risco_id": rid,
            "candidatos_cobertos": SEPARADOR.join(c["candidato_id"] for c in marcados),
            "cobertura": derivar_cobertura([c["origem"] for c in marcados]),
            "justificativa": trocar_codigos(justificativa, por_codigo),
            **esperado,
        }
        conf.gravar_tabela(caminho, JULGAMENTO_FIELDS, [linha], acrescentar=True)
        saida(f"[ok] {pid} / {rid}: {linha['cobertura']} ({len(marcados)} candidato(s))")
    return falharam


def julgar_riscos_ia(
    corpus: Path,
    provider: AIProvider | None,
    *,
    modelo: str,
    limite: int | None = None,
    dry_run: bool = False,
    saida: Callable[[str], None] = print,
) -> int:
    """Classifica os riscos de IA dos processos pendentes; devolve quantos processos falharam."""
    corpus, saida_dir, todos, processos, candidatos = _preparar(corpus, modelo, limite)
    caminho = saida_dir / RISCOS_IA_CSV
    esperado = {"modelo_juiz": modelo, "versao_prompt_juiz": VERSAO_RISCOS_IA}
    feitos = {linha["processo_id"] for linha in _ja_gravadas(caminho, RISCOS_IA_FIELDS, esperado)}
    pendentes = [p for p in processos if p.processo_id not in feitos]
    saida(
        f"{len(todos)} processo(s) incluido(s); {len(processos)} nesta execucao; "
        f"{len(processos) - len(pendentes)} ja classificado(s); {len(pendentes)} a classificar (1 chamada cada)."
    )
    saida(f"Juiz {modelo} | prompt {VERSAO_RISCOS_IA} | saida: {caminho}")
    if dry_run:
        saida("[dry-run] nada gravado.")
        return 0
    if provider is None:
        raise JulgamentoError("Modelo juiz nao configurado.")

    falharam = seguidas = 0
    for posicao, processo in enumerate(pendentes, start=1):
        pid = processo.processo_id
        de_ia = [c for c in candidatos[pid] if c["origem"] == ORIGIN_AI]
        codificados = [(f"R{ordem:02d}", c) for ordem, c in enumerate(de_ia, start=1)]
        codigos = [codigo for codigo, _ in codificados]
        try:
            if not de_ia:
                raise RespostaInvalida("processo sem riscos de IA entre os candidatos")
            resposta = _perguntar(
                provider, prompt_riscos_ia(processo, codificados), lambda texto: interpretar_riscos_ia(texto, codigos)
            )
        except (AIProviderError, RespostaInvalida) as exc:
            falharam += 1
            seguidas += 1
            saida(f"[FALHA] {pid}: {gerar._resumir_erro(str(exc))}")
            if seguidas >= FALHAS_SEGUIDAS_MAX:
                restantes = len(pendentes) - posicao
                falharam += restantes
                saida(f"{seguidas} falhas seguidas; execucao interrompida ({restantes} processo(s) nem foram tentados).")
                break
            continue
        seguidas = 0
        linhas = [
            {
                "processo_id": pid,
                "candidato_id": c["candidato_id"],
                "categoria": c["categoria"],
                "nomeia_objeto": resposta[codigo]["nomeia_objeto"],
                "termo": SEPARADOR.join(resposta[codigo]["termos"]),
                "especifico_do_objeto": resposta[codigo]["especifico_do_objeto"],
                "grupo": resposta[codigo]["grupo"],
                **esperado,
                "conferir": conferir_termo(c, resposta[codigo]["nomeia_objeto"], resposta[codigo]["termos"]),
            }
            for codigo, c in codificados
        ]
        conf.gravar_tabela(caminho, RISCOS_IA_FIELDS, linhas, acrescentar=True)  # o processo inteiro de uma vez
        nomeiam = sum(1 for linha in linhas if linha["nomeia_objeto"] == "sim")
        especificos = sum(1 for linha in linhas if linha["especifico_do_objeto"] == "sim")
        grupos = len({linha["grupo"] for linha in linhas})
        a_conferir = sum(1 for linha in linhas if linha["conferir"] != "nao")
        saida(
            f"[ok] {pid}: {len(linhas)} risco(s) de IA; {nomeiam} nomeiam o objeto; {especificos} especificos; "
            f"{grupos} grupo(s); {a_conferir} a conferir."
        )
    return falharam


def construir_juiz(timeout: float = TIMEOUT_PADRAO) -> tuple[AIProvider, str, str]:
    """``(provedor, nome do provedor, modelo)`` a partir de JUDGE_PROVIDER/JUDGE_MODEL."""
    provedor = (ai_provider._config_value("JUDGE_PROVIDER") or "").strip().lower()
    modelo = (ai_provider._config_value("JUDGE_MODEL") or "").strip()
    if not provedor or not modelo:
        raise JulgamentoError("Defina JUDGE_PROVIDER e JUDGE_MODEL (variavel de ambiente ou .streamlit/secrets.toml).")
    config = ai_provider.load_ai_config()
    if modelo == config.model:
        raise JulgamentoError(f"JUDGE_MODEL ({modelo}) tem de ser diferente do modelo gerador (AI_MODEL).")
    juiz = dataclasses.replace(config, provider=provedor, model=modelo)
    # temperatura 0 e fixada em src/ai_provider.py (TEMPERATURE), que so a envia quando o SDK aceita o
    # parametro; a API da Anthropic o recusa nos modelos atuais ("temperature: Extra inputs are not permitted")
    provider = ai_provider.build_provider(juiz, timeout=timeout, max_retries=NOVAS_TENTATIVAS)
    if hasattr(provider, "max_tokens"):
        # o raciocinio do modelo conta no limite de saida; com o padrao (2048) o JSON chegava cortado
        provider.max_tokens = MAX_TOKENS_JUIZ
    return provider, provedor, modelo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", required=True, type=Path, help="Pasta do corpus.")
    parser.add_argument("--riscos-ia", action="store_true", help="Classificar os riscos de IA (nomeia_objeto, especifico_do_objeto, grupo).")
    parser.add_argument("--limite", type=int, help="Processar so os N primeiros processos incluidos.")
    parser.add_argument("--semente", type=int, default=SEMENTE_PADRAO, help="Semente do embaralhamento (cobertura).")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_PADRAO, help="Timeout de cada chamada, em segundos.")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar o que seria feito, sem chamar o juiz.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"Pasta do corpus nao encontrada: {corpus}", file=sys.stderr)
        return 2
    os.chdir(ROOT)  # .streamlit/secrets.toml e lido a partir da raiz do projeto
    try:
        provider, provedor, modelo = construir_juiz(args.timeout)
        print(f"Provedor do juiz: {provedor}")
        if args.riscos_ia:
            falharam = julgar_riscos_ia(corpus, provider, modelo=modelo, limite=args.limite, dry_run=args.dry_run)
        else:
            falharam = julgar_cobertura(
                corpus, provider, modelo=modelo, semente=args.semente, limite=args.limite, dry_run=args.dry_run
            )
    except (JulgamentoError, conf.ConferenciaError, AIProviderError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 1 if falharam else 0


if __name__ == "__main__":
    raise SystemExit(main())
