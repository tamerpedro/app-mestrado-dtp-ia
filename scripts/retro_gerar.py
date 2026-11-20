"""Avaliacao retrospectiva, Parte 1: casos e geracao em lote dos riscos de IA.

Para cada processo com ``incluir=sim`` em ``<corpus>/corpus_casos.csv`` cria um
caso no formato de ``data/casos/`` e gera os riscos de IA com as mesmas
funcoes, o mesmo prompt e o mesmo provedor que o app usa. Nada e gravado no
repositorio: tudo vai para ``<corpus>/retrospectiva/``.

- ``casos/<processo_id>.json``: o caso. Os mapas publicados nao trazem valor,
  prazo, criticidade nem modalidade, entao todos os casos usam os mesmos
  valores neutros (``VALORES_NEUTROS``); o objeto vai so no campo ``objeto``.
- ``casos/<processo_id>_ia_pt.json``: cache dos riscos de IA. So e gravado
  quando todas as categorias vieram completas e nunca e sobrescrito.
- ``geracao_log.csv``: uma linha por caso em que a IA foi chamada.
- ``candidatos.csv``: para cada processo ja gerado, os riscos do catalogo
  aplicaveis ao tipo de contratacao e os riscos de IA do caso.

``corpus_casos.csv`` e ``data/riscos_base.csv`` sao apenas lidos.

Uso:
    python scripts/retro_gerar.py --corpus "<pasta do corpus>" [--limite N] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for _caminho in (ROOT, SCRIPTS):
    if str(_caminho) not in sys.path:
        sys.path.insert(0, str(_caminho))

import retro_conferencia as conf  # noqa: E402
from src.ai_provider import AIProvider, AIProviderError, build_provider, load_ai_config  # noqa: E402
from src.case_store import ai_cache_path, load_cached_ai_risks, load_case, save_cached_ai_risks  # noqa: E402
from src.catalog_mix import (  # noqa: E402
    ORIGIN_AI,
    ORIGIN_CATALOG,
    ai_count_for_category,
    filter_catalog_by_contract_type,
    generate_ai_risks_for_all_categories,
)
from src.domain import CATEGORIES, CONTRACT_TYPES  # noqa: E402
from src.eval_mode import library_hash  # noqa: E402
from src.models import RiskItem  # noqa: E402
from src.risk_generator import PROMPT_VERSION_V2, GeneratedRisk  # noqa: E402
from src.risk_library import load_risks  # noqa: E402

BIBLIOTECA = ROOT / "data" / "riscos_base.csv"
CASOS_CSV = "corpus_casos.csv"
RISCOS_CSV = "corpus_riscos_reais.csv"
SAIDA_DIR = "retrospectiva"
CASOS_DIR = "casos"
LOG_CSV = "geracao_log.csv"
CANDIDATOS_CSV = "candidatos.csv"
IDIOMA = "pt"
TENTATIVAS_PADRAO = 3

# tipo_sugerido do corpus -> codigo de src/domain.py (decisao do autor: licenciamento = software).
TIPOS = {"aquisicao": "aquisicao", "servico": "servico", "licenciamento": "software"}

# Iguais para todos os casos: o mapa publicado nao informa nenhum destes campos.
VALORES_NEUTROS = {"valor_estimado": 0, "criticidade": "media", "prazo": "", "modalidade": "", "contexto": ""}

LOG_FIELDS = [
    "processo_id",
    "provedor",
    "modelo",
    "versao_prompt",
    "hash_biblioteca",
    "data_hora",
    "situacao",
    "tentativas",
    *(f"qtd_{categoria}" for categoria in CATEGORIES),
    "qtd_total",
    "falhas",
]
CANDIDATOS_FIELDS = [
    "processo_id",
    "candidato_id",
    "origem",
    "id_catalogo",
    "categoria",
    "titulo",
    "causa",
    "consequencia",
]


class GeracaoError(Exception):
    """Falha que impede preparar os casos ou gerar os riscos."""


@dataclass(frozen=True)
class Processo:
    processo_id: str
    objeto: str
    tipo_contratacao: str  # codigo de domain.CONTRACT_TYPES


# --------------------------------------------------------------------------
# Leitura do corpus e validacao
# --------------------------------------------------------------------------


def ler_processos(corpus: Path) -> list[Processo]:
    """Processos com ``incluir=sim``, validados (objeto, tipo e riscos reais)."""
    try:
        linhas = conf.ler_tabela(corpus / CASOS_CSV, ["processo_id", "objeto", "tipo_sugerido", "incluir"])
        riscos = conf.ler_tabela(corpus / RISCOS_CSV, ["processo_id", "risco"])
    except conf.ConferenciaError as exc:
        raise GeracaoError(str(exc)) from exc
    com_riscos = {risco["processo_id"] for risco in riscos if risco["risco"]}
    incluidos = [linha for linha in linhas if conf._sim_nao(linha["incluir"]) == "sim"]

    problemas: list[str] = []
    vistos: set[str] = set()
    processos: list[Processo] = []
    for linha in incluidos:
        processo_id = linha["processo_id"]
        tipo = conf._sim_nao(linha["tipo_sugerido"])
        if not processo_id:
            problemas.append("linha com incluir=sim sem processo_id")
            continue
        if processo_id in vistos:
            problemas.append(f"{processo_id}: processo_id repetido")
        vistos.add(processo_id)
        if not linha["objeto"]:
            problemas.append(f"{processo_id}: objeto vazio")
        if processo_id not in com_riscos:
            problemas.append(f"{processo_id}: sem riscos em {RISCOS_CSV}")
        if tipo not in TIPOS:
            problemas.append(
                f"{processo_id}: tipo_sugerido '{linha['tipo_sugerido']}' sem codigo correspondente "
                f"({', '.join(CONTRACT_TYPES)}); defina o codigo antes de gerar"
            )
            continue
        processos.append(Processo(processo_id, linha["objeto"], TIPOS[tipo]))
    if problemas:
        raise GeracaoError("Corpus com problema; nada foi gerado:\n" + "\n".join(f"  - {p}" for p in problemas))
    if not processos:
        raise GeracaoError(f"Nenhum processo com incluir=sim em {CASOS_CSV}.")
    return processos


# --------------------------------------------------------------------------
# Casos
# --------------------------------------------------------------------------


def conteudo_do_caso(processo: Processo) -> str:
    """JSON do caso, no formato de ``data/casos/``."""
    caso = {
        "nome": processo.processo_id,
        "objeto": processo.objeto,
        "tipo_contratacao": processo.tipo_contratacao,
        "valor_estimado": VALORES_NEUTROS["valor_estimado"],
        "criticidade": VALORES_NEUTROS["criticidade"],
        "prazo": VALORES_NEUTROS["prazo"],
        "modalidade": VALORES_NEUTROS["modalidade"],
        "contexto": VALORES_NEUTROS["contexto"],
    }
    return json.dumps(caso, ensure_ascii=False, indent=2) + "\n"


def situacao_do_caso(casos_dir: Path, processo: Processo) -> str:
    """``novo``, ``igual``, ``mudou`` (sem cache) ou ``mudou_com_cache``."""
    caminho = casos_dir / f"{processo.processo_id}.json"
    if not caminho.exists():
        return "novo"
    if caminho.read_text(encoding="utf-8") == conteudo_do_caso(processo):
        return "igual"
    return "mudou_com_cache" if ai_cache_path(caminho, IDIOMA, casos_dir).exists() else "mudou"


def gravar_caso(casos_dir: Path, processo: Processo) -> Path:
    caminho = casos_dir / f"{processo.processo_id}.json"
    casos_dir.mkdir(parents=True, exist_ok=True)
    caminho.write_text(conteudo_do_caso(processo), encoding="utf-8")
    return caminho


# --------------------------------------------------------------------------
# Geracao
# --------------------------------------------------------------------------


def quantidades_esperadas(biblioteca: list[RiskItem]) -> dict[str, int]:
    """Quantos riscos de IA o app pede por categoria (``min(3, n)`` do catalogo)."""
    return {
        categoria: ai_count_for_category(sum(1 for risco in biblioteca if risco.categoria == categoria))
        for categoria in CATEGORIES
    }


_REQUEST_ID_RE = re.compile(r",?\s*'request_id':\s*'[^']*'")


def _resumir_erro(erro: str) -> str:
    """Erro em uma linha, sem o identificador da requisicao (muda a cada chamada)."""
    return " ".join(_REQUEST_ID_RE.sub("", erro).split())


def gerar_caso(
    caso_path: Path,
    casos_dir: Path,
    biblioteca: list[RiskItem],
    provider: AIProvider,
    tentativas: int,
) -> tuple[dict[str, list[GeneratedRisk]] | None, int, list[str], bool]:
    """Gera os riscos de IA do caso; grava o cache so se todas as categorias vierem completas.

    Devolve ``(riscos ou None, tentativas feitas, falhas, recusa geral)``. Usa as
    funcoes do projeto (``generate_ai_risks_for_all_categories`` e
    ``save_cached_ai_risks``), as mesmas chamadas por ``get_or_generate_case_ai_risks``.

    Recusa geral: todas as categorias falharam com o mesmo erro (chave, saldo,
    rede). Nao adianta insistir; o chamador interrompe a execucao.
    """
    _, contexto = load_case(caso_path)
    esperado = quantidades_esperadas(biblioteca)
    pedidas = [categoria for categoria, quantidade in esperado.items() if quantidade]
    falhas: list[str] = []
    for tentativa in range(1, tentativas + 1):
        erros: dict[str, str] = {}
        gerados = generate_ai_risks_for_all_categories(
            biblioteca,
            contexto,
            provider,
            lang=IDIOMA,
            on_ai_failure=lambda categoria, erro: erros.__setitem__(categoria, _resumir_erro(erro)),
        )
        for categoria, quantidade in esperado.items():
            obtido = len(gerados.get(categoria, []))
            if obtido != quantidade and categoria not in erros:
                erros[categoria] = f"{obtido} risco(s), esperado {quantidade}"
        if not erros:
            if ai_cache_path(caso_path, IDIOMA, casos_dir).exists():  # nunca sobrescrever
                raise GeracaoError(f"{caso_path.stem}: o cache apareceu durante a geracao; nada foi sobrescrito.")
            save_cached_ai_risks(caso_path, IDIOMA, gerados, casos_dir)
            return gerados, tentativa, falhas, False
        if set(erros) == set(pedidas) and len(set(erros.values())) == 1:
            falhas.append(f"tentativa {tentativa}: todas as categorias: {next(iter(erros.values()))}")
            return None, tentativa, falhas, True
        falhas.extend(f"tentativa {tentativa}: {categoria}: {erro}" for categoria, erro in erros.items())
    return None, tentativas, falhas, False


# --------------------------------------------------------------------------
# Candidatos
# --------------------------------------------------------------------------


def linhas_de_candidatos(
    processo: Processo, biblioteca: list[RiskItem], riscos_ia: dict[str, list[GeneratedRisk]]
) -> list[dict]:
    """Catalogo aplicavel ao tipo de contratacao + riscos de IA do caso."""
    linhas = []
    for risco in filter_catalog_by_contract_type(biblioteca, processo.tipo_contratacao):
        linhas.append(
            {
                "processo_id": processo.processo_id,
                "candidato_id": f"{processo.processo_id}_cat_{risco.id}",
                "origem": ORIGIN_CATALOG,
                "id_catalogo": risco.id,
                "categoria": risco.categoria,
                "titulo": risco.text("titulo", IDIOMA),
                "causa": risco.text("causa", IDIOMA),
                "consequencia": risco.text("consequencia", IDIOMA),
            }
        )
    for categoria in CATEGORIES:
        for ordem, gerado in enumerate(riscos_ia.get(categoria, []), start=1):
            linhas.append(
                {
                    "processo_id": processo.processo_id,
                    "candidato_id": f"{processo.processo_id}_ia_{categoria}_{ordem}",
                    "origem": ORIGIN_AI,
                    "id_catalogo": "",
                    "categoria": categoria,
                    "titulo": gerado.titulo,
                    "causa": gerado.causa,
                    "consequencia": gerado.consequencia,
                }
            )
    return [{campo: " ".join(str(valor).split()) for campo, valor in linha.items()} for linha in linhas]


def gravar_candidatos(saida_dir: Path, processos: list[Processo], biblioteca: list[RiskItem]) -> tuple[int, list[str]]:
    """Regrava ``candidatos.csv`` com os processos que ja tem cache; devolve (linhas, sem cache)."""
    casos_dir = saida_dir / CASOS_DIR
    linhas: list[dict] = []
    sem_cache: list[str] = []
    for processo in processos:
        caso_path = casos_dir / f"{processo.processo_id}.json"
        riscos_ia = load_cached_ai_risks(caso_path, IDIOMA, casos_dir) if caso_path.exists() else None
        if riscos_ia is None:
            sem_cache.append(processo.processo_id)
            continue
        linhas.extend(linhas_de_candidatos(processo, biblioteca, riscos_ia))
    conf.gravar_tabela(saida_dir / CANDIDATOS_CSV, CANDIDATOS_FIELDS, linhas)
    return len(linhas), sem_cache


# --------------------------------------------------------------------------
# Execucao
# --------------------------------------------------------------------------


def executar(
    corpus: Path,
    provider: AIProvider | None,
    *,
    provedor: str,
    modelo: str,
    limite: int | None = None,
    dry_run: bool = False,
    tentativas: int = TENTATIVAS_PADRAO,
    biblioteca_path: Path = BIBLIOTECA,
    agora: Callable[[], datetime] = datetime.now,
    saida: Callable[[str], None] = print,
) -> int:
    """Prepara os casos e gera os riscos de IA pendentes; devolve quantos casos falharam."""
    corpus = corpus.resolve()
    if corpus == ROOT or ROOT in corpus.parents:
        raise GeracaoError("A pasta do corpus nao pode ficar dentro do repositorio.")
    todos = ler_processos(corpus)
    processos = todos[:limite] if limite else todos
    biblioteca = load_risks(biblioteca_path)
    hash_biblioteca = library_hash(biblioteca_path)
    esperado = quantidades_esperadas(biblioteca)
    saida_dir = corpus / SAIDA_DIR
    casos_dir = saida_dir / CASOS_DIR

    situacoes = {processo.processo_id: situacao_do_caso(casos_dir, processo) for processo in processos}
    conflitos = [pid for pid, situacao in situacoes.items() if situacao == "mudou_com_cache"]
    if conflitos:
        raise GeracaoError(
            "O objeto ou o tipo mudou em processo que ja tem cache de IA (gerado para o caso anterior): "
            f"{', '.join(conflitos)}. O cache nunca e sobrescrito; apague-o a mao se quiser gerar de novo."
        )
    pendentes = [
        p for p in processos if not ai_cache_path(casos_dir / f"{p.processo_id}.json", IDIOMA, casos_dir).exists()
    ]
    chamadas = sum(1 for quantidade in esperado.values() if quantidade)
    saida(
        f"{len(todos)} processo(s) incluido(s); {len(processos)} nesta execucao; "
        f"{len(processos) - len(pendentes)} ja com cache; {len(pendentes)} a gerar."
    )
    saida(
        f"Por caso: {chamadas} chamada(s) de IA e {sum(esperado.values())} risco(s) "
        f"({', '.join(f'{c}={q}' for c, q in esperado.items())})."
    )
    saida(f"Prompt {PROMPT_VERSION_V2} | {provedor}/{modelo} | biblioteca {hash_biblioteca[:12]} | saida: {saida_dir}")

    if dry_run:
        for processo in processos:
            pendente = "gerar" if processo in pendentes else "cache existe"
            saida(
                f"[dry-run] {processo.processo_id}: caso {situacoes[processo.processo_id]}; {pendente}; "
                f"tipo={processo.tipo_contratacao}; catalogo aplicavel="
                f"{len(filter_catalog_by_contract_type(biblioteca, processo.tipo_contratacao))}; objeto={processo.objeto}"
            )
        saida(f"[dry-run] nada gravado; {len(pendentes) * chamadas} chamada(s) de IA seriam feitas.")
        return 0

    if provider is None:
        raise GeracaoError("Provedor de IA nao configurado.")
    for processo in processos:
        if situacoes[processo.processo_id] != "igual":
            gravar_caso(casos_dir, processo)

    falharam = 0
    for processo in pendentes:
        caso_path = casos_dir / f"{processo.processo_id}.json"
        gerados, feitas, falhas, recusa_geral = gerar_caso(caso_path, casos_dir, biblioteca, provider, tentativas)
        quantidades = {c: len((gerados or {}).get(c, [])) for c in CATEGORIES}
        registro = {
            "processo_id": processo.processo_id,
            "provedor": provedor,
            "modelo": modelo,
            "versao_prompt": PROMPT_VERSION_V2,
            "hash_biblioteca": hash_biblioteca,
            "data_hora": agora().isoformat(timespec="seconds"),
            "situacao": "gerado" if gerados is not None else "falhou",
            "tentativas": feitas,
            **{f"qtd_{categoria}": quantidade for categoria, quantidade in quantidades.items()},
            "qtd_total": sum(quantidades.values()),
            "falhas": " | ".join(falhas),
        }
        conf.gravar_tabela(saida_dir / LOG_CSV, LOG_FIELDS, [registro], acrescentar=True)
        if gerados is None:
            falharam += 1
            saida(f"[FALHA] {processo.processo_id}: {feitas} tentativa(s) sem as categorias completas; sem cache.")
            if recusa_geral:
                restantes = len(pendentes) - pendentes.index(processo) - 1
                saida(
                    "O provedor recusou todas as chamadas com o mesmo erro; execucao interrompida "
                    f"({restantes} caso(s) nem foram tentados): {falhas[-1].split(': ', 2)[-1]}"
                )
                break
        else:
            saida(f"[ok] {processo.processo_id}: {registro['qtd_total']} risco(s) de IA em {feitas} tentativa(s).")

    total, sem_cache = gravar_candidatos(saida_dir, todos, biblioteca)
    saida(f"{CANDIDATOS_CSV}: {total} linha(s); {len(todos) - len(sem_cache)} processo(s) com cache.")
    if sem_cache:
        saida(f"Ainda sem cache (fora de {CANDIDATOS_CSV}): {', '.join(sem_cache)}")
    return falharam


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", required=True, type=Path, help="Pasta do corpus.")
    parser.add_argument("--limite", type=int, help="Processar so os N primeiros casos incluidos.")
    parser.add_argument("--dry-run", action="store_true", help="Validar e mostrar o que seria feito, sem chamar a IA.")
    parser.add_argument("--tentativas", type=int, default=TENTATIVAS_PADRAO, help="Tentativas por caso.")
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
        provider = None if args.dry_run else build_provider(config)  # mesmos padroes do app
        falharam = executar(
            corpus,
            provider,
            provedor=config.provider,
            modelo=config.model,
            limite=args.limite,
            dry_run=args.dry_run,
            tentativas=args.tentativas,
        )
    except (GeracaoError, conf.ConferenciaError, AIProviderError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 1 if falharam else 0


if __name__ == "__main__":
    raise SystemExit(main())
