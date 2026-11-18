"""Amostra de conferencia reduzida (``retro_amostra.py --reduzida``). Sem rede e sem modelo."""

import csv
import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def _carregar(nome):
    if nome in sys.modules:
        return sys.modules[nome]
    spec = importlib.util.spec_from_file_location(nome, SCRIPTS / f"{nome}.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


_carregar("retro_extrair")
_carregar("retro_conferencia")
_carregar("retro_gerar")
_carregar("retro_julgar")
amostra = _carregar("retro_amostra")

CASOS = "processo_id;grupo;orgao;ano;arquivo;paginas_mapa;objeto;tipo_sugerido;objeto_confirmado;incluir;analise_principal;observacao"
RISCOS = "processo_id;risco_id;fase;risco;causa;consequencia;probabilidade;impacto;pagina;conferir"
CANDIDATOS = "processo_id;candidato_id;origem;id_catalogo;categoria;titulo;causa;consequencia"
JULGAMENTO = "processo_id;risco_id;candidatos_cobertos;cobertura;justificativa;modelo_juiz;versao_prompt_juiz;semente"
PROCESSOS = tuple(f"P{n:02d}-01" for n in range(1, 7))
RISCOS_POR_PROCESSO = 9  # 54 riscos -> 40 na amostra
CANDIDATOS_POR_PROCESSO = 20
ORIGINAIS = ("amostra_conferencia.csv", "amostra_chave.csv", "julgamento.csv", "candidatos.csv")


def _gravar(caminho, cabecalho, linhas):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(("\r\n".join([cabecalho, *linhas]) + "\r\n").encode("utf-8-sig"))


def _ler(caminho):
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


def _candidato_id(pid, k):
    return f"{pid}_cat_R{k:03d}" if k <= 12 else f"{pid}_ia_gestao_{k}"


def _marcados(pid, n):
    """O que o 'juiz' marcou no risco n: nada, 2 candidatos ou 11 (mais que os 9 da lista)."""
    quantos = {0: 11, 2: 2}.get(n % 3, 0)
    return [_candidato_id(pid, k) for k in range(1, quantos + 1)]


@pytest.fixture
def corpus(tmp_path):
    """6 processos, 9 riscos reais e 20 candidatos cada, ja julgados e com a amostra atual sorteada."""
    pasta = tmp_path / "corpus"
    _gravar(pasta / "corpus_casos.csv", CASOS, [f"{pid};externo;Órgão;2024;{pid}.pdf;1;Objeto;aquisicao;sim;sim;sim;" for pid in PROCESSOS])
    reais, candidatos, julgamento = [], [], []
    for pid in PROCESSOS:
        for n in range(1, RISCOS_POR_PROCESSO + 1):
            reais.append(f"{pid};R{n};planejamento;Atraso no tema{n};Falta de alfa{n};Prejuízo beta{n};Baixa;Alta;1;nao")
            julgamento.append(f"{pid};R{n};{' | '.join(_marcados(pid, n))};ambos;Motivo reservado.;modelo-x;v1;20261117")
        for k in range(1, CANDIDATOS_POR_PROCESSO + 1):
            origem, categoria = ("catalogo", "planejamento") if k <= 12 else ("ia", "gestao")
            tema = k % RISCOS_POR_PROCESSO + 1
            candidatos.append(
                f"{pid};{_candidato_id(pid, k)};{origem};{f'R{k:03d}' if k <= 12 else ''};{categoria};"
                f"Evento {k} no tema{tema};Origem alfa{tema} item{k};Efeito gama{k}"
            )
    _gravar(pasta / "corpus_riscos_reais.csv", RISCOS, reais)
    _gravar(pasta / "retrospectiva" / "candidatos.csv", CANDIDATOS, candidatos)
    _gravar(pasta / "retrospectiva" / "julgamento.csv", JULGAMENTO, julgamento)
    assert amostra.gerar_amostra(pasta, saida=lambda _m: None) == 40
    return pasta


def _por_risco(linhas):
    grupos = {}
    for linha in linhas:
        grupos.setdefault((linha["processo_id"], linha["risco_id"]), []).append(linha)
    return grupos


# ---------------------------------------------------------------- similaridade


def test_similaridade_tfidf_ignora_acentos_e_palavras_gramaticais():
    valores = amostra.similaridades(
        "Atraso na entrega da solução",
        ["atraso de entrega para solucao", "Indisponibilidade do datacenter", "Atraso no pagamento", ""],
    )
    assert valores[0] == pytest.approx(1.0)
    assert valores[1] == 0.0 and valores[3] == 0.0
    assert 0.0 < valores[2] < valores[0]
    assert amostra.similaridades("", ["qualquer texto"]) == [0.0]


def test_ranking_usa_risco_causa_e_consequencia_e_desempata_pela_ordem():
    risco = {"risco": "Atraso na entrega", "causa": "Fornecedor sem estoque", "consequencia": "Multa contratual"}
    candidatos = [
        {"candidato_id": "a", "titulo": "Vazamento de dados", "causa": "Falha de acesso", "consequencia": "Sanção"},
        {"candidato_id": "b", "titulo": "Entrega fora do prazo", "causa": "Estoque do fornecedor", "consequencia": "Multa"},
        {"candidato_id": "c", "titulo": "Preço acima do mercado", "causa": "Pesquisa ruim", "consequencia": "Sobrepreço"},
        {"candidato_id": "d", "titulo": "Atraso", "causa": "", "consequencia": ""},
    ]
    ranking = amostra.ranking_de_similaridade(risco, candidatos)
    assert sorted(posicao for posicao, _ in ranking.values()) == [1, 2, 3, 4]
    assert ranking["b"][0] == 1 and ranking["d"][0] == 2
    assert ranking["a"] == (3, 0.0) and ranking["c"] == (4, 0.0)  # empate: ordem do arquivo


# ---------------------------------------------------------------- composicao da lista


@pytest.mark.parametrize(("marcados", "do_juiz", "parecidos"), [(0, 0, 9), (2, 2, 7), (9, 9, 0), (11, 11, 0)])
def test_lista_tem_marcados_mais_parecidos_ate_nove_e_tres_sorteados(marcados, do_juiz, parecidos):
    candidatos = [{"candidato_id": f"c{k:02d}"} for k in range(1, 31)]
    ranking = {f"c{k:02d}": (31 - k, k / 100) for k in range(1, 31)}  # c30 e o mais parecido
    ids_marcados = {f"c{k:02d}" for k in range(1, marcados + 1)}  # os MENOS parecidos: entram mesmo assim
    lista = amostra.compor_lista(candidatos, ids_marcados, ranking, random.Random(1))

    ids = [c["candidato_id"] for c, _ in lista]
    assert len(ids) == len(set(ids)) == max(marcados, 9) + 3
    entradas = Counter(entrada for _, entrada in lista)
    assert entradas == Counter({"juiz": do_juiz, "similaridade": parecidos, "sorteio": 3}) - Counter()
    assert {c["candidato_id"] for c, e in lista if e == "juiz"} == ids_marcados
    assert [c["candidato_id"] for c, e in lista if e == "similaridade"] == [f"c{k:02d}" for k in range(30, 30 - parecidos, -1)]
    # os sorteados vem do que sobrou: nem marcados, nem entre os mais parecidos
    sobra = {f"c{k:02d}" for k in range(marcados + 1, 31 - parecidos)}
    assert {c["candidato_id"] for c, e in lista if e == "sorteio"} <= sobra


def test_lista_com_poucos_candidatos_nao_repete():
    candidatos = [{"candidato_id": f"c{k}"} for k in range(1, 11)]
    ranking = {f"c{k}": (k, 0.0) for k in range(1, 11)}
    lista = amostra.compor_lista(candidatos, {"c10"}, ranking, random.Random(1))
    assert sorted(c["candidato_id"] for c, _ in lista) == sorted(c["candidato_id"] for c in candidatos)
    assert Counter(e for _, e in lista) == {"juiz": 1, "similaridade": 8, "sorteio": 1}


def test_listas_completas_sao_cinco_de_pelo_menos_quatro_processos():
    riscos = [("A", f"R{n}") for n in range(1, 41)] + [("B", "R1"), ("C", "R1"), ("D", "R1")]  # um processo domina
    for semente in range(30):
        escolhidos = amostra.sortear_listas_completas(riscos, semente)
        assert len(escolhidos) == 5 and escolhidos <= set(riscos)
        assert len({pid for pid, _ in escolhidos}) >= 4
        assert escolhidos == amostra.sortear_listas_completas(riscos, semente)
    assert amostra.sortear_listas_completas(riscos[:3], 1) == set(riscos[:3])  # menos riscos que o pedido


# ---------------------------------------------------------------- arquivo gerado


def test_reduzida_preserva_os_riscos_e_compoe_as_listas(corpus):
    saida_dir = corpus / "retrospectiva"
    antes = {nome: (saida_dir / nome).read_bytes() for nome in ORIGINAIS}
    mensagens = []
    assert amostra.gerar_amostra_reduzida(corpus, saida=mensagens.append) == 40
    assert {nome: (saida_dir / nome).read_bytes() for nome in ORIGINAIS} == antes  # nada regravado

    atual = _por_risco(_ler(saida_dir / "amostra_conferencia.csv"))
    linhas = _ler(saida_dir / "amostra_conferencia_reduzida.csv")
    chave = _ler(saida_dir / "amostra_chave_reduzida.csv")
    reduzida, chave_por_risco = _por_risco(linhas), _por_risco(chave)
    assert (saida_dir / "amostra_conferencia_reduzida.csv").read_bytes().startswith(b"\xef\xbb\xbfprocesso_id;risco_id;")
    assert list(linhas[0]) == amostra.AMOSTRA_FIELDS and list(chave[0]) == amostra.CHAVE_REDUZIDA_FIELDS
    assert set(reduzida) == set(atual) and len(reduzida) == 40  # os mesmos riscos
    assert len(linhas) == len(chave)
    assert all(l["cobre_autor"] == "" and l["observacao"] == "" for l in linhas)
    assert any(f"{len(linhas)} linha(s)" in m and "5 com a lista completa" in m for m in mensagens)

    # as 5 listas completas ficam no fim, vem de pelo menos 4 processos e trazem todos os candidatos
    ordem = list(reduzida)
    completas = [r for r in ordem if chave_por_risco[r][0]["entrada"] == "lista completa"]
    assert completas == ordem[-5:] and len({pid for pid, _ in completas}) >= 4
    codigos_da_chave_atual = {(c["processo_id"], c["candidato_id"]): c["candidato_codigo"] for c in _ler(saida_dir / "amostra_chave.csv")}
    textos_atuais = {(l["processo_id"], l["candidato_id"]): l["titulo_candidato"] for l in _ler(saida_dir / "amostra_conferencia.csv")}
    candidatos = {c["candidato_id"]: c for c in _ler(saida_dir / "candidatos.csv")}
    tamanhos = Counter()
    for (pid, rid), do_risco in reduzida.items():
        da_chave = chave_por_risco[(pid, rid)]
        codigos = [l["candidato_id"] for l in do_risco]
        assert len(codigos) == len(set(codigos))  # sem repeticao
        assert codigos == [c["candidato_codigo"] for c in da_chave]  # a chave acompanha a amostra, linha a linha
        assert do_risco[0]["risco_real"] == atual[(pid, rid)][0]["risco_real"]
        for linha, c in zip(do_risco, da_chave):
            # mesmo codigo e mesmo texto da amostra atual; a chave devolve o candidato original
            assert codigos_da_chave_atual[(pid, c["candidato_id"])] == linha["candidato_id"]
            assert textos_atuais[(pid, linha["candidato_id"])] == linha["titulo_candidato"] == candidatos[c["candidato_id"]]["titulo"]
            assert linha["causa_candidato"] == candidatos[c["candidato_id"]]["causa"]
            assert linha["consequencia_candidato"] == candidatos[c["candidato_id"]]["consequencia"]
        posicoes = sorted(int(c["posicao_similaridade"]) for c in da_chave)
        entradas = Counter(c["entrada"] for c in da_chave)
        marcados = set(_marcados(pid, int(rid[1:])))
        if (pid, rid) in completas:
            assert entradas == {"lista completa": CANDIDATOS_POR_PROCESSO}
            assert posicoes == list(range(1, CANDIDATOS_POR_PROCESSO + 1))
            continue
        tamanhos[len(marcados)] += 1
        assert len(do_risco) == max(len(marcados), 9) + 3
        assert {c["candidato_id"] for c in da_chave if c["entrada"] == "juiz"} == marcados
        assert entradas["similaridade"] == max(0, 9 - len(marcados)) and entradas["sorteio"] == 3
        # os de similaridade sao os mais bem colocados entre os nao marcados
        fora = [int(c["posicao_similaridade"]) for c in da_chave if c["entrada"] == "sorteio"]
        dentro = [int(c["posicao_similaridade"]) for c in da_chave if c["entrada"] == "similaridade"]
        assert not dentro or max(dentro) < min(fora)
        de_fato = amostra.ranking_de_similaridade(
            {"risco": f"Atraso no tema{rid[1:]}", "causa": f"Falta de alfa{rid[1:]}", "consequencia": f"Prejuízo beta{rid[1:]}"},
            [c for c in candidatos.values() if c["processo_id"] == pid],
        )
        assert all(de_fato[c["candidato_id"]][0] == int(c["posicao_similaridade"]) for c in da_chave)
        nao_marcados = sorted(posicao for cid, (posicao, _) in de_fato.items() if cid not in marcados)
        assert sorted(dentro) == nao_marcados[: len(dentro)]
    assert set(tamanhos) == {0, 2, 11}  # os tres casos apareceram entre as listas reduzidas
    assert len(linhas) == 5 * CANDIDATOS_POR_PROCESSO + tamanhos[0] * 12 + tamanhos[2] * 12 + tamanhos[11] * 14


def test_reduzida_nao_revela_origem_nem_juiz_e_embaralha_cada_risco(corpus):
    saida_dir = corpus / "retrospectiva"
    amostra.gerar_amostra_reduzida(corpus, saida=lambda _m: None)
    bruto = (saida_dir / "amostra_conferencia_reduzida.csv").read_text(encoding="utf-8-sig")
    for vazamento in ("_cat_", "_ia_", "catalogo", "ambos", "juiz", "modelo-x", "Motivo", "similaridade", "sorteio", "lista completa"):
        assert vazamento not in bruto

    chave = _por_risco(_ler(saida_dir / "amostra_chave_reduzida.csv"))
    reduzidas = [linhas for linhas in chave.values() if linhas[0]["entrada"] != "lista completa"]
    # a posicao na lista nao indica o motivo da entrada: os marcados nao ficam sempre no comeco
    com_marcados = [linhas for linhas in reduzidas if any(l["entrada"] == "juiz" for l in linhas)]
    assert any(linhas[0]["entrada"] != "juiz" for linhas in com_marcados)
    assert any(
        [l["entrada"] for l in linhas] != sorted((l["entrada"] for l in linhas), key=["juiz", "similaridade", "sorteio"].index)
        for linhas in reduzidas
    )
    # nem a ordem dos codigos: riscos do mesmo processo nao repetem a mesma sequencia
    completas = [[l["candidato_codigo"] for l in linhas] for linhas in chave.values() if linhas[0]["entrada"] == "lista completa"]
    assert all(codigos != sorted(codigos) for codigos in completas)


def test_reduzida_e_reprodutivel_e_nao_sobrescreve(corpus):
    saida_dir = corpus / "retrospectiva"
    nomes = ("amostra_conferencia_reduzida.csv", "amostra_chave_reduzida.csv")
    amostra.gerar_amostra_reduzida(corpus, saida=lambda _m: None)
    primeira = [(saida_dir / nome).read_bytes() for nome in nomes]

    with pytest.raises(amostra.AmostraError, match="ja existe"):
        amostra.gerar_amostra_reduzida(corpus, saida=lambda _m: None)
    assert [(saida_dir / nome).read_bytes() for nome in nomes] == primeira

    for nome in nomes:
        (saida_dir / nome).unlink()
    assert amostra.main(["--corpus", str(corpus), "--reduzida"]) == 0
    assert [(saida_dir / nome).read_bytes() for nome in nomes] == primeira


def test_reduzida_exige_a_amostra_atual(tmp_path, corpus):
    (corpus / "retrospectiva" / "amostra_chave.csv").unlink()
    with pytest.raises(amostra.AmostraError, match="amostra_chave.csv nao encontrado"):
        amostra.gerar_amostra_reduzida(corpus, saida=lambda _m: None)
    assert not (corpus / "retrospectiva" / "amostra_conferencia_reduzida.csv").exists()
