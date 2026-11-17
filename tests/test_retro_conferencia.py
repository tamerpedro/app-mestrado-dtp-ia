import csv
import importlib.util
import io
import math
import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _carregar(nome):
    if nome in sys.modules:
        return sys.modules[nome]
    spec = importlib.util.spec_from_file_location(nome, SCRIPTS / f"{nome}.py")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome] = modulo
    spec.loader.exec_module(modulo)
    return modulo


retro = _carregar("retro_extrair")
conf = _carregar("retro_conferencia")

CASOS_CABECALHO = "processo_id;grupo;orgao;ano;arquivo;paginas_mapa;objeto;tipo_sugerido;objeto_confirmado;incluir;observacao"


def _risco(pid, rid, risco, causa="", consequencia="", pagina=1, conferir="nao"):
    return {
        "processo_id": pid,
        "risco_id": rid,
        "fase": "planejamento",
        "risco": risco,
        "causa": causa,
        "consequencia": consequencia,
        "probabilidade": "Baixa",
        "impacto": "Alto",
        "pagina": str(pagina),
        "conferir": conferir,
    }


def _corpus(tmp_path, riscos, paginas="1 a 3"):
    corpus = tmp_path / "corpus"
    (corpus / "pdfs").mkdir(parents=True)
    processos = sorted({r["processo_id"] for r in riscos})
    casos = [CASOS_CABECALHO] + [f"{p};externo;Orgao;2024;{p}.pdf;{paginas};Objeto;servico;sim;sim;" for p in processos]
    (corpus / "corpus_casos.csv").write_bytes(("\r\n".join(casos) + "\r\n").encode("utf-8-sig"))
    conf.gravar_tabela(corpus / "corpus_riscos_reais.csv", retro.RISCOS_FIELDS, riscos)
    return corpus


def _leitor(textos_por_processo, palavras_por_processo=None):
    """Paginas fixas por processo: {processo: {numero: texto}}."""

    def ler(pdf_path, inicial, final):
        processo = pdf_path.stem
        textos = textos_por_processo.get(processo, {})
        palavras = (palavras_por_processo or {}).get(processo, {})
        return [
            retro.Pagina(
                numero=n,
                texto=textos.get(n, ""),
                palavras=[retro.Palavra(texto=t, x0=0, x1=10) for t in palavras.get(n, [])],
            )
            for n in range(inicial, final + 1)
        ]

    return ler


def _ler(corpus, nome):
    with (corpus / nome).open(encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


def _gerar(corpus, leitor, **kwargs):
    return conf.gerar(corpus, leitor=leitor, saida=lambda _m: None, **kwargs)


def _aplicar(corpus):
    mensagens = []
    resumo = conf.aplicar(corpus, agora=datetime(2026, 11, 20, 10, 0, 0), saida=mensagens.append)
    return resumo, mensagens


def _preencher(corpus, respostas, *, encoding="utf-8-sig", sep=";"):
    """respostas: {(processo, risco_id, campo): (ok_autor, texto_corrigido)}; o resto fica 'sim'."""
    linhas = _ler(corpus, "conferencia_extracao.csv")
    for linha in linhas:
        if linha["processo_id"].startswith("#"):
            continue
        ok, texto = respostas.get((linha["processo_id"], linha["risco_id"], linha["campo"]), ("sim", ""))
        linha["ok_autor"], linha["texto_corrigido"] = ok, texto
    saida = io.StringIO(newline="")
    escritor = csv.writer(saida, delimiter=sep)
    escritor.writerow(conf.CONFERENCIA_FIELDS)
    for linha in linhas:
        escritor.writerow([linha[campo] or "" for campo in conf.CONFERENCIA_FIELDS])
    (corpus / "conferencia_extracao.csv").write_bytes(saida.getvalue().encode(encoding))


# ---------------------------------------------------------------- --gerar


def test_gerar_inclui_conferir_sim_e_campos_nao_encontrados(tmp_path):
    riscos = [
        _risco("AAA-01", "R1", "Falha na pesquisa de precos", "Pressa da equipe", "Atraso na entrega | Sobrepreco relevante"),
        _risco("AAA-01", "R2", "Risco fora do texto da pagina", conferir="sim"),
        _risco("AAA-01", "R3", "Licitacao deserta ou fracassada", "Causa inventada pelo modelo", "Atraso na contratacao", pagina=2),
        _risco("AAA-01", "R4", "Fiscalizacao deficiente do contrato", "", "Baixa qualidade | Dano que nao esta na pagina", pagina=2),
        _risco("AAA-01", "R5", "Descumprimento de clausulas", "", "Dano que so aparece na pagina seguinte", pagina=2),
    ]
    paginas = {
        "AAA-01": {
            1: "Falha na pesquisa de precos. Pressa da equipe. Atraso na entrega. Sobrepreco relevante",
            2: "Licitacao deserta ou fracassada; Atraso na contratacao; Fiscalizacao deficiente do contrato; "
            "Baixa qualidade; Descumprimento de clausulas",
            3: "Dano que so aparece na pagina seguinte",
        }
    }
    corpus = _corpus(tmp_path, riscos)

    linhas = _gerar(corpus, _leitor(paginas))

    obrigatorias = [(l["risco_id"], l["campo"], l["motivo"]) for l in linhas if l["motivo"] != "amostra"]
    assert obrigatorias == [
        ("R2", "risco", "enunciado nao encontrado"),
        ("R3", "causa", "causa nao encontrada"),
        ("R4", "consequencia", "consequencia nao encontrada"),
        ("R5", "consequencia", "consequencia nao encontrada"),
    ]
    r4 = next(l for l in linhas if l["risco_id"] == "R4")
    assert r4["texto_extraido"] == "Baixa qualidade | Dano que nao esta na pagina"  # o campo inteiro
    # so R1 restou sem pendencia: entra na amostra, uma linha por campo preenchido
    assert [(l["risco_id"], l["campo"]) for l in linhas if l["motivo"] == "amostra"] == [
        ("R1", "risco"),
        ("R1", "causa"),
        ("R1", "consequencia"),
    ]


def test_gerar_grava_csv_com_bom_colunas_e_semente(tmp_path):
    corpus = _corpus(tmp_path, [_risco("AAA-01", "R1", "Falha na pesquisa de precos")])
    _gerar(corpus, _leitor({"AAA-01": {1: "Falha na pesquisa de precos"}}), semente=77)

    bruto = (corpus / "conferencia_extracao.csv").read_bytes()
    assert bruto.startswith(b"\xef\xbb\xbf" + ";".join(conf.CONFERENCIA_FIELDS).encode())
    linhas = _ler(corpus, "conferencia_extracao.csv")
    assert all(l["ok_autor"] == l["texto_corrigido"] == l["observacao"] == "" for l in linhas)
    metadado = linhas[-1]
    assert metadado["processo_id"] == "#" and metadado["campo"] == "semente" and metadado["texto_extraido"] == "77"


def test_gerar_nao_sobrescreve_arquivo_existente(tmp_path):
    corpus = _corpus(tmp_path, [_risco("AAA-01", "R1", "Falha na pesquisa de precos")])
    leitor = _leitor({"AAA-01": {1: "Falha na pesquisa de precos"}})
    _gerar(corpus, leitor)
    _preencher(corpus, {})
    preenchido = (corpus / "conferencia_extracao.csv").read_bytes()
    with pytest.raises(conf.ConferenciaError, match="ja existe"):
        _gerar(corpus, leitor)
    assert (corpus / "conferencia_extracao.csv").read_bytes() == preenchido
    _gerar(corpus, leitor, sobrescrever=True)
    assert (corpus / "conferencia_extracao.csv").read_bytes() != preenchido


def _muitos(quantidades):
    riscos, paginas = [], {}
    for processo, quantidade in quantidades.items():
        textos = []
        for i in range(quantidade):
            enunciado = f"Risco numero {i:03d} do processo {processo}"
            riscos.append(_risco(processo, f"R{i:03d}", enunciado, pagina=1 + i % 3))
            textos.append(enunciado)
        paginas[processo] = {n: " ; ".join(textos) for n in (1, 2, 3)}
    return riscos, paginas


@pytest.mark.parametrize(
    ("quantidades", "esperado"),
    [
        ({"AAA-01": 150, "BBB-01": 148, "CCC-01": 2}, math.ceil(0.15 * 300)),  # 15%
        ({"AAA-01": 50, "BBB-01": 9, "CCC-01": 1}, 30),  # minimo de 30
        ({"AAA-01": 12, "BBB-01": 5}, 17),  # menos de 30 restantes: todos
    ],
)
def test_amostra_tem_tamanho_certo_e_ao_menos_um_por_processo(tmp_path, quantidades, esperado):
    riscos, paginas = _muitos(quantidades)
    corpus = _corpus(tmp_path, riscos)
    linhas = _gerar(corpus, _leitor(paginas))
    sorteados = {(l["processo_id"], l["risco_id"]) for l in linhas}
    assert {l["motivo"] for l in linhas} == {"amostra"}
    assert len(sorteados) == esperado == len(linhas)  # so o campo risco esta preenchido
    assert {processo for processo, _ in sorteados} == set(quantidades)


def test_amostra_e_reprodutivel_pela_semente(tmp_path):
    riscos, paginas = _muitos({"AAA-01": 120, "BBB-01": 80})
    corpus = _corpus(tmp_path, riscos)
    ids = lambda linhas: [(l["processo_id"], l["risco_id"]) for l in linhas]  # noqa: E731
    primeira = ids(_gerar(corpus, _leitor(paginas), semente=1))
    assert ids(_gerar(corpus, _leitor(paginas), semente=1, sobrescrever=True)) == primeira
    assert ids(_gerar(corpus, _leitor(paginas), semente=2, sobrescrever=True)) != primeira


def test_arquivo_ordenado_por_processo_e_pagina(tmp_path):
    riscos = [
        _risco("BBB-01", "R1", "Enunciado ausente um", pagina=3, conferir="sim"),
        _risco("AAA-01", "R9", "Enunciado ausente dois", pagina=2, conferir="sim"),
        _risco("BBB-01", "R2", "Enunciado ausente tres", pagina=1, conferir="sim"),
        _risco("AAA-01", "R3", "Enunciado ausente quatro", pagina=1, conferir="sim"),
    ]
    corpus = _corpus(tmp_path, riscos)
    linhas = _gerar(corpus, _leitor({}))
    assert [(l["processo_id"], l["pagina"], l["risco_id"]) for l in linhas] == [
        ("AAA-01", "1", "R3"),
        ("AAA-01", "2", "R9"),
        ("BBB-01", "1", "R2"),
        ("BBB-01", "3", "R1"),
    ]


def test_sugere_texto_com_ti_restaurado_igual_nos_dois_processos(tmp_path):
    enunciado = "Par cipação de empresas sem os obje vos ins tucionais man das."
    riscos = [
        _risco(p, "4", enunciado, "", "Contratação que não atende aos obje vos", conferir="sim")
        for p in ("UFV-01", "UFV-02")
    ] + [_risco("UFV-01", "5", "Risco sem lacuna, mas ausente", conferir="sim")]
    palavras = {1: ["Par\x00cipação", "obje\x00vos", "ins\x00tucionais", "man\x00das.", "\x00"]}
    corpus = _corpus(tmp_path, riscos)

    linhas = _gerar(corpus, _leitor({}, {"UFV-01": palavras, "UFV-02": palavras}))

    sugestoes = {(l["processo_id"], l["risco_id"], l["campo"]): l["texto_sugerido"] for l in linhas}
    certo = "Participação de empresas sem os objetivos institucionais mantidas."
    assert sugestoes[("UFV-01", "4", "risco")] == sugestoes[("UFV-02", "4", "risco")] == certo
    assert sugestoes[("UFV-01", "4", "consequencia")] == "Contratação que não atende aos objetivos"
    assert sugestoes[("UFV-01", "5", "risco")] == ""  # sem lacuna, sem sugestao
    # a sugestao nao e aplicada: a extracao continua como estava
    assert _ler(corpus, "corpus_riscos_reais.csv")[0]["risco"] == enunciado


def test_restaurar_ti_so_troca_a_palavra_inteira():
    formas = ["ob\x00da"]
    assert conf.restaurar_ti("proposta ob da acima", formas) == "proposta obtida acima"
    assert conf.restaurar_ti("sob dados reais", formas) == "sob dados reais"


# ---------------------------------------------------------------- --aplicar


@pytest.fixture
def conferencia(tmp_path):
    riscos = [
        _risco("AAA-01", "R1", "Falha na pesquisa de precos", "Pressa da equipe", "Atraso na obra"),
        _risco("AAA-01", "R2", "Risco com lacuna no enunciado", "", "Dano com lacuna", conferir="sim"),
        _risco("AAA-01", "R3", "Enunciado que estava certo afinal", conferir="sim"),
        _risco("BBB-01", "R1", "Licitacao deserta ou fracassada", "", "Retrabalho da equipe"),
    ]
    paginas = {
        "AAA-01": {1: "Falha na pesquisa de precos; Pressa da equipe; Atraso na obra"},
        "BBB-01": {1: "Licitacao deserta ou fracassada; Retrabalho da equipe"},
    }
    corpus = _corpus(tmp_path, riscos)
    _gerar(corpus, _leitor(paginas))
    return corpus


RESPOSTAS = {
    ("AAA-01", "R2", "risco"): ("nao", "Risco  sem lacuna\nno enunciado"),
    ("AAA-01", "R1", "causa"): ("Não", "Pressa da equipe de planejamento"),
}


def test_aplicar_corrige_so_o_campo_e_marca_conferir(conferencia):
    antes = (conferencia / "corpus_riscos_reais.csv").read_bytes()
    casos = (conferencia / "corpus_casos.csv").read_bytes()
    _preencher(conferencia, RESPOSTAS)

    resumo, _ = _aplicar(conferencia)

    assert (conferencia / "corpus_riscos_reais_original.csv").read_bytes() == antes
    assert (conferencia / "corpus_casos.csv").read_bytes() == casos
    depois = {(r["processo_id"], r["risco_id"]): r for r in _ler(conferencia, "corpus_riscos_reais.csv")}
    original = {(r["processo_id"], r["risco_id"]): r for r in _ler(conferencia, "corpus_riscos_reais_original.csv")}
    # R2: enunciado corrigido (em uma linha); o resto do risco igual
    assert depois[("AAA-01", "R2")] == {
        **original[("AAA-01", "R2")],
        "risco": "Risco sem lacuna no enunciado",
        "conferir": "corrigido",
    }
    # R1: causa corrigida em risco que estava com conferir=nao
    assert depois[("AAA-01", "R1")] == {
        **original[("AAA-01", "R1")],
        "causa": "Pressa da equipe de planejamento",
        "conferir": "corrigido",
    }
    # R3: ok_autor=sim em risco com conferir=sim vira conferido, sem mudar texto
    assert depois[("AAA-01", "R3")] == {**original[("AAA-01", "R3")], "conferir": "conferido"}
    # BBB R1: ok_autor=sim em risco com conferir=nao nao muda nada
    assert depois[("BBB-01", "R1")] == original[("BBB-01", "R1")]
    assert (conferencia / "corpus_riscos_reais.csv").read_bytes().startswith(b"\xef\xbb\xbf")

    registro = _ler(conferencia, "correcoes_extracao.csv")
    assert list(registro[0]) == conf.CORRECOES_FIELDS
    assert [(r["risco_id"], r["campo"], r["texto_antes"], r["texto_depois"], r["motivo"]) for r in registro] == [
        ("R1", "causa", "Pressa da equipe", "Pressa da equipe de planejamento", "amostra"),
        ("R2", "risco", "Risco com lacuna no enunciado", "Risco sem lacuna no enunciado", "enunciado nao encontrado"),
        ("R3", "conferir", "sim", "conferido", "enunciado nao encontrado"),
    ]
    assert {r["data_hora"] for r in registro} == {"2026-11-20T10:00:00"}
    assert resumo["corrigidos"] == 2 and resumo["alteracoes_gravadas"] == 3
    assert resumo["por_processo"]["AAA-01"]["corrigidos"] == 2
    assert resumo["por_processo"]["BBB-01"] == {"confirmados": 2}


def test_aplicar_calcula_taxa_de_erro_so_da_amostra(conferencia):
    _preencher(conferencia, RESPOSTAS)
    resumo, mensagens = _aplicar(conferencia)
    # amostra: R1 (risco, causa, consequencia) de AAA e R1 (risco, consequencia) de BBB; 1 erro
    assert (resumo["amostra_linhas"], resumo["amostra_erros"]) == (5, 1)
    assert resumo["amostra_taxa_erro"] == pytest.approx(0.2)
    assert any("1/5" in m and "20.0%" in m for m in mensagens)


def test_aplicar_de_novo_nao_duplica_nem_troca_o_original(conferencia):
    _preencher(conferencia, RESPOSTAS)
    _aplicar(conferencia)
    original = (conferencia / "corpus_riscos_reais_original.csv").read_bytes()
    riscos = (conferencia / "corpus_riscos_reais.csv").read_bytes()
    registro = (conferencia / "correcoes_extracao.csv").read_bytes()

    resumo, _ = _aplicar(conferencia)

    assert resumo["alteracoes_gravadas"] == 0 and resumo["corrigidos"] == 2
    assert (conferencia / "corpus_riscos_reais_original.csv").read_bytes() == original
    assert (conferencia / "corpus_riscos_reais.csv").read_bytes() == riscos
    assert (conferencia / "correcoes_extracao.csv").read_bytes() == registro


@pytest.mark.parametrize(
    ("respostas", "mensagem"),
    [
        ({("AAA-01", "R3", "risco"): ("", "")}, "ok_autor vazio"),
        ({("AAA-01", "R3", "risco"): ("talvez", "")}, "ok_autor vazio"),
        ({("AAA-01", "R3", "risco"): ("nao", "  ")}, "sem texto_corrigido"),
        ({("AAA-01", "R3", "risco"): ("sim", "Texto que nao deveria estar aqui")}, "qual dos dois"),
    ],
)
def test_aplicar_para_sem_alterar_nada_se_o_preenchimento_esta_incompleto(conferencia, respostas, mensagem):
    antes = (conferencia / "corpus_riscos_reais.csv").read_bytes()
    _preencher(conferencia, {**RESPOSTAS, **respostas})
    with pytest.raises(conf.ConferenciaError, match=mensagem) as erro:
        _aplicar(conferencia)
    assert "AAA-01 R3 risco" in str(erro.value)
    assert (conferencia / "corpus_riscos_reais.csv").read_bytes() == antes
    assert not (conferencia / "corpus_riscos_reais_original.csv").exists()
    assert not (conferencia / "correcoes_extracao.csv").exists()


def test_aplicar_para_se_a_extracao_mudou_depois_do_gerar(conferencia):
    _preencher(conferencia, RESPOSTAS)
    riscos = _ler(conferencia, "corpus_riscos_reais.csv")
    riscos[3]["risco"] = "Enunciado refeito por outra extracao"
    conf.gravar_tabela(conferencia / "corpus_riscos_reais.csv", retro.RISCOS_FIELDS, riscos)
    with pytest.raises(conf.ConferenciaError, match="texto_extraido nao confere"):
        _aplicar(conferencia)


@pytest.mark.parametrize(("encoding", "sep"), [("cp1252", "\t"), ("utf-16", "\t"), ("utf-8", ";")])
def test_aplicar_le_arquivo_salvo_pelo_excel(conferencia, encoding, sep):
    respostas = {
        ("AAA-01", "R2", "risco"): ("NÃO", "Risco sem lacuna no enunciado, com acentuação"),
        ("AAA-01", "R3", "risco"): (" Sim ", ""),
    }
    _preencher(conferencia, respostas, encoding=encoding, sep=sep)
    resumo, _ = _aplicar(conferencia)
    depois = {(r["processo_id"], r["risco_id"]): r for r in _ler(conferencia, "corpus_riscos_reais.csv")}
    assert depois[("AAA-01", "R2")]["risco"] == "Risco sem lacuna no enunciado, com acentuação"
    assert depois[("AAA-01", "R3")]["conferir"] == "conferido"
    assert resumo["corrigidos"] == 1


# ---------------------------------------------------------------- [VAZIO] e regra do nivel de impacto


def test_vazio_limpa_o_campo_e_registra_texto_depois_vazio(conferencia):
    _preencher(conferencia, {("AAA-01", "R1", "consequencia"): ("nao", " [VAZIO] ")})
    resumo, mensagens = _aplicar(conferencia)

    depois = {(r["processo_id"], r["risco_id"]): r for r in _ler(conferencia, "corpus_riscos_reais.csv")}
    assert depois[("AAA-01", "R1")]["consequencia"] == ""
    assert depois[("AAA-01", "R1")]["causa"] == "Pressa da equipe"  # so o campo pedido
    assert depois[("AAA-01", "R1")]["conferir"] == "corrigido"
    [linha] = [r for r in _ler(conferencia, "correcoes_extracao.csv") if r["campo"] == "consequencia"]
    assert (linha["texto_antes"], linha["texto_depois"], linha["motivo"]) == ("Atraso na obra", "", "amostra")
    # campo limpo na amostra conta como erro de campo trocado, nao de transcricao
    assert (resumo["amostra_erros"], resumo["amostra_erros_campo_trocado"], resumo["amostra_erros_transcricao"]) == (1, 1, 0)
    assert any("campo trocado" in m and "1/5" in m for m in mensagens)

    riscos = (conferencia / "corpus_riscos_reais.csv").read_bytes()
    registro = (conferencia / "correcoes_extracao.csv").read_bytes()
    assert _aplicar(conferencia)[0]["alteracoes_gravadas"] == 0  # de novo: nada muda
    assert (conferencia / "corpus_riscos_reais.csv").read_bytes() == riscos
    assert (conferencia / "correcoes_extracao.csv").read_bytes() == registro


def test_amostra_separa_transcricao_de_campo_trocado(conferencia):
    _preencher(
        conferencia,
        {
            ("AAA-01", "R1", "causa"): ("nao", "Pressa da equipe de planejamento"),
            ("BBB-01", "R1", "consequencia"): ("nao", "[VAZIO]"),
        },
    )
    resumo, _ = _aplicar(conferencia)
    assert resumo["amostra_linhas"] == 5 and resumo["amostra_riscos"] == 2
    assert (resumo["amostra_erros_transcricao"], resumo["amostra_erros_campo_trocado"]) == (1, 1)
    assert resumo["amostra_taxa_transcricao"] == pytest.approx(0.2)
    assert resumo["amostra_taxa_erro"] == pytest.approx(0.4)


def test_vazio_nao_vale_para_o_enunciado(conferencia):
    _preencher(conferencia, {("AAA-01", "R3", "risco"): ("nao", "[VAZIO]")})
    with pytest.raises(conf.ConferenciaError, match="nao vale para o enunciado"):
        _aplicar(conferencia)


@pytest.fixture
def ibram(tmp_path):
    riscos = [
        _risco("IBRAM-01", "R-01", "Nao aprovacao dos artefatos", "Atraso na contratacao", "Alto"),
        _risco("IBRAM-01", "R-04", "Atraso no processo administrativo", "Demora", "alto"),
        _risco("IBRAM-01", "R-05", "Atraso ou suspensao da licitacao", "Impugnacoes", "Muito Alto"),
        _risco("IBRAM-01", "R-08", "Nao manter a proposta", "Desistencia", "Médio"),
        _risco("IBRAM-01", "R-12", "Gestao contratual inadequada", "", "Responsavel sem competencia | Alto risco de ma mensuracao"),
        _risco("OUTRO-01", "R1", "Risco de outro orgao qualquer", "", "Alto"),
    ]
    paginas = {
        "IBRAM-01": {
            1: "Nao aprovacao dos artefatos; Atraso na contratacao; Atraso no processo administrativo; Demora; "
            "Atraso ou suspensao da licitacao; Impugnacoes; Muito Alto; Nao manter a proposta; Desistencia; "
            "Gestao contratual inadequada; Responsavel sem competencia; Alto risco de ma mensuracao"
        },
        "OUTRO-01": {1: "Risco de outro orgao qualquer"},
    }
    corpus = _corpus(tmp_path, riscos)
    _gerar(corpus, _leitor(paginas))
    return corpus


def test_regra_limpa_nivel_de_impacto_gravado_como_consequencia(ibram):
    # o autor confirma que a transcricao e fiel (sim) ou marca nao sem texto: a regra resolve os dois casos
    _preencher(ibram, {("IBRAM-01", "R-05", "consequencia"): ("nao", "")})
    resumo, _ = _aplicar(ibram)

    depois = {(r["processo_id"], r["risco_id"]): r for r in _ler(ibram, "corpus_riscos_reais.csv")}
    original = {(r["processo_id"], r["risco_id"]): r for r in _ler(ibram, "corpus_riscos_reais_original.csv")}
    for rid in ("R-01", "R-04", "R-05", "R-08"):
        assert depois[("IBRAM-01", rid)] == {**original[("IBRAM-01", rid)], "consequencia": "", "conferir": "corrigido"}
    assert depois[("IBRAM-01", "R-12")] == original[("IBRAM-01", "R-12")]  # tem texto: fica
    assert depois[("OUTRO-01", "R1")] == original[("OUTRO-01", "R1")]  # regra so vale para o IBRAM-01

    registro = _ler(ibram, "correcoes_extracao.csv")
    assert [(r["risco_id"], r["campo"], r["texto_antes"], r["texto_depois"], r["motivo"]) for r in registro] == [
        ("R-01", "consequencia", "Alto", "", "nivel de impacto gravado como consequencia"),
        ("R-04", "consequencia", "alto", "", "nivel de impacto gravado como consequencia"),
        ("R-05", "consequencia", "Muito Alto", "", "nivel de impacto gravado como consequencia"),
        ("R-08", "consequencia", "Médio", "", "nivel de impacto gravado como consequencia"),
    ]
    assert resumo["limpos_pela_regra"] == 4
    assert resumo["por_processo"]["IBRAM-01"]["limpos_pela_regra"] == 4
    assert "corrigidos" not in resumo["por_processo"]["IBRAM-01"]  # R-05 conta na regra, nao duas vezes

    riscos = (ibram / "corpus_riscos_reais.csv").read_bytes()
    log = (ibram / "correcoes_extracao.csv").read_bytes()
    de_novo, _ = _aplicar(ibram)  # de novo: nada muda e o resumo se mantem
    assert de_novo["alteracoes_gravadas"] == 0 and de_novo["limpos_pela_regra"] == 4
    assert (ibram / "corpus_riscos_reais.csv").read_bytes() == riscos
    assert (ibram / "correcoes_extracao.csv").read_bytes() == log


def test_nao_sem_texto_continua_sendo_erro_fora_da_regra(ibram):
    _preencher(ibram, {("IBRAM-01", "R-12", "consequencia"): ("nao", "")})
    with pytest.raises(conf.ConferenciaError, match=r"sem texto_corrigido \(use \[VAZIO\]"):
        _aplicar(ibram)
    assert not (ibram / "correcoes_extracao.csv").exists()