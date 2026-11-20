import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.ai_provider import FakeProvider

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "retro_extrair.py"
_spec = importlib.util.spec_from_file_location("retro_extrair", MODULE_PATH)
retro = importlib.util.module_from_spec(_spec)
sys.modules["retro_extrair"] = retro
_spec.loader.exec_module(retro)

CABECALHO = "processo_id;grupo;orgao;ano;arquivo;paginas_mapa;objeto;tipo_sugerido;objeto_confirmado;incluir;observacao"


def _linha(pid, paginas="1 a 2", incluir="sim", observacao=""):
    return f"{pid};externo;Orgao;2024;{pid}.pdf;{paginas};Objeto;servico;sim;{incluir};{observacao}"


def _corpus(tmp_path, linhas, *, encoding="utf-8-sig", sep=";"):
    corpus = tmp_path / "corpus"
    (corpus / "pdfs").mkdir(parents=True)
    texto = "\r\n".join([CABECALHO, *linhas]).replace(";", sep) + "\r\n"
    (corpus / "corpus_casos.csv").write_bytes(texto.encode(encoding))
    for linha in linhas:
        (corpus / "pdfs" / f"{linha.split(';')[0]}.pdf").write_bytes(b"")
    return corpus


def _resposta(*riscos):
    return json.dumps({"riscos": list(riscos)}, ensure_ascii=False)


def _risco(texto, pagina=1, **extra):
    return {"risco_id": "R01", "risco": texto, "pagina": pagina, **extra}


class LeitorFalso:
    """Substitui a leitura do PDF: registra o intervalo pedido e devolve paginas fixas."""

    def __init__(self, texto="Risco: Falha na pesquisa de precos\nDano: atraso"):
        self.texto = texto
        self.chamadas = []

    def __call__(self, pdf_path, inicial, final):
        self.chamadas.append((pdf_path.name, inicial, final))
        return [retro.Pagina(numero=n, texto=f"{self.texto} (pagina {n})") for n in range(inicial, final + 1)]


def _executar(corpus, provider, leitor, **kwargs):
    mensagens = []
    falhas = retro.executar(
        corpus, provider, provedor="fake", modelo="modelo-fake", leitor=leitor, saida=mensagens.append, **kwargs
    )
    return falhas, mensagens


def _saida(corpus, nome):
    with (corpus / nome).open(encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


# ---------------------------------------------------------------- JSON


def test_resposta_valida_normaliza_campos():
    bruto = _resposta(
        {
            "risco_id": "R-02",
            "risco": "Falha  na\npesquisa de precos",
            "causa": None,
            "consequencia": ["Atraso na contratacao", "", "Sobrepreco"],
            "fase": "planejamento",
            "probabilidade": 2,
            "impacto": "4 - Alto",
            "pagina": "3",
        }
    )
    [risco] = retro.interpretar_resposta(bruto)
    assert risco == {
        "risco_id": "R-02",
        "fase": "planejamento",
        "risco": "Falha na pesquisa de precos",
        "causa": "",
        "consequencia": "Atraso na contratacao | Sobrepreco",
        "probabilidade": "2",
        "impacto": "4 - Alto",
        "pagina": 3,
    }


def test_resposta_com_cerca_de_codigo_e_sem_id_usa_ordem_de_aparicao():
    bruto = "```json\n" + _resposta({"risco": "Primeiro risco"}, {"risco": "Segundo risco"}) + "\n```"
    riscos = retro.interpretar_resposta(bruto)
    assert [r["risco_id"] for r in riscos] == ["1", "2"]
    assert riscos[0]["pagina"] is None


def test_lista_vazia_e_resposta_valida():
    assert retro.interpretar_resposta('{"riscos": []}') == []


@pytest.mark.parametrize(
    "bruto",
    [
        "",
        "nao sei",
        '{"riscos": [{"risco": "cortado no meio',
        '{"itens": []}',
        '{"riscos": "nenhum"}',
        '{"riscos": ["texto solto"]}',
        '{"riscos": [{"risco_id": "R01", "risco": "  "}]}',
    ],
)
def test_resposta_invalida_levanta_erro(bruto):
    with pytest.raises(retro.ExtracaoError):
        retro.interpretar_resposta(bruto)


# ---------------------------------------------------------------- checagem literal


def test_literal_ignora_quebra_de_linha_espacos_e_hifenizacao():
    pagina = retro.Pagina(numero=1, texto="Risco 01:  Atraso na contra-\ntação  do  serviço de\nimpressão on-site.")
    assert retro.localizar_trecho("Atraso na contratação do serviço de impressão on-site", pagina) == "texto"
    assert retro.localizar_trecho("Atraso na contra- tação do serviço", pagina) == "texto"


def test_literal_nao_aceita_texto_resumido_ou_corrigido():
    pagina = retro.Pagina(numero=1, texto="Risco 01: Atrazo na contratação do serviço de impressão")
    assert retro.localizar_trecho("Atraso na contratação do serviço de impressão", pagina) is None
    assert retro.localizar_trecho("Atrazo na contratação de impressão", pagina) is None


def test_literal_trecho_curto_demais_nunca_casa():
    assert retro.localizar_trecho("Risco", retro.Pagina(numero=1, texto="Risco 01: Atraso")) is None


def test_literal_acha_em_celula_de_tabela():
    intercalado = "Falha na Agentes sem\npesquisa de precos capacitacao"
    em_tabela = retro.Pagina(
        numero=1, texto=intercalado, tabelas=[[["1", "Falha na\npesquisa de precos", "Agentes sem\ncapacitacao"]]]
    )
    assert retro.localizar_trecho("Falha na pesquisa de precos", em_tabela) == "tabela"
    assert retro.localizar_trecho("Falha na pesquisa de precos", retro.Pagina(numero=1, texto=intercalado)) is None


def _palavras(*linhas):
    """Cada linha: lista de (texto, x0). Largura de 6 pontos por caractere."""
    return [retro.Palavra(texto=t, x0=x, x1=x + 6 * len(t)) for linha in linhas for t, x in linha]


def test_literal_acha_em_coluna_de_tabela_sem_bordas():
    # Coluna "Risco" em x=100..200 e coluna "Causa" em x=300..; linhas intercaladas.
    palavras = _palavras(
        [("Definição", 100), ("de", 160), ("Utilização", 300), ("de", 370), ("especificações", 390)],
        [("especificações", 100), ("técnicas", 300), ("não", 360)],
        [("R-01", 20), ("incompletas", 100), ("ou", 172), ("padronizadas", 300)],
        [("restritivas", 100), ("de", 300), ("outros", 320)],
    )
    pagina = retro.Pagina(numero=1, texto="(intercalado)", palavras=palavras)
    assert retro.localizar_trecho("Definição de especificações incompletas ou restritivas", pagina) == "coluna"
    assert retro.localizar_trecho("Utilização de especificações técnicas não padronizadas de outros", pagina) == "coluna"
    # Pular palavra da propria coluna e resumir, nao transcrever.
    assert retro.localizar_trecho("Definição de especificações restritivas", pagina) is None


def test_literal_acha_enunciado_que_atravessa_a_pagina():
    pagina = retro.Pagina(numero=1, texto="Risco 07: Descumprimento de cláusulas")
    proxima = retro.Pagina(numero=2, texto="contratuais pela contratada\nDano 1: ...")
    trecho = "Descumprimento de cláusulas contratuais pela contratada"
    assert retro.localizar_trecho(trecho, pagina) is None
    assert retro.localizar_trecho(trecho, pagina, proxima) == "texto+proxima"


def test_marcar_conferir_usa_a_pagina_indicada():
    paginas = [
        retro.Pagina(numero=4, texto="Risco: Falha na caracterização do objeto"),
        retro.Pagina(numero=5, texto="Risco: Restrição da competitividade"),
    ]
    riscos = [
        {"risco": "Falha na caracterização do objeto", "pagina": 4},
        {"risco": "Restrição da competitividade", "pagina": 4},  # esta na pagina 5
        {"risco": "Risco que o modelo inventou agora", "pagina": 5},
        {"risco": "Falha na caracterização do objeto", "pagina": None},
        {"risco": "Falha na caracterização do objeto", "pagina": 9},  # fora do intervalo
    ]
    retro.marcar_conferir(riscos, paginas)
    assert [r["conferir"] for r in riscos] == ["nao", "sim", "sim", "sim", "sim"]


# ---------------------------------------------------------------- corpus_casos.csv


def test_le_csv_com_ponto_e_virgula_e_bom(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01", "29 a 40", observacao="Usar só a seção 2")])
    [caso] = retro.ler_casos(corpus / "corpus_casos.csv")
    assert caso == retro.Caso("AAA-01", "AAA-01.pdf", 29, 40, "Usar só a seção 2")


def test_le_csv_com_tabulacao_e_windows_1252(tmp_path):
    linhas = [
        _linha("AAA-01", incluir="SIM", observacao="Seção única"),
        _linha("BBB-01", incluir="não"),
        _linha("CCC-01", incluir="Nao"),
        _linha("DDD-01", incluir=" Sim "),
        _linha("EEE-01", incluir=""),
    ]
    corpus = _corpus(tmp_path, linhas, encoding="cp1252", sep="\t")
    casos = retro.ler_casos(corpus / "corpus_casos.csv")
    assert [caso.processo_id for caso in casos] == ["AAA-01", "DDD-01"]
    assert casos[0].observacao == "Seção única"


@pytest.mark.parametrize("valor", ["01/mai", "1-5", "", "5 a 2", "0 a 3", "45778"])
def test_paginas_mapa_fora_do_formato_para_tudo(tmp_path, valor):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01", valor)])
    with pytest.raises(retro.ExtracaoError, match="BBB-01"):
        retro.ler_casos(corpus / "corpus_casos.csv")
    provider = FakeProvider(_resposta())
    with pytest.raises(retro.ExtracaoError):
        _executar(corpus, provider, LeitorFalso())
    assert provider.calls == []  # nem o processo valido e extraido antes do aviso


def test_paginas_mapa_invalido_em_linha_excluida_nao_impede(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01", "01/mai", incluir="nao")])
    assert [caso.processo_id for caso in retro.ler_casos(corpus / "corpus_casos.csv")] == ["AAA-01"]


# ---------------------------------------------------------------- execucao


def test_so_processa_incluir_sim_e_so_o_intervalo_de_paginas(tmp_path):
    corpus = _corpus(
        tmp_path,
        [_linha("AAA-01", "3 a 4", observacao="Usar so a secao 2"), _linha("BBB-01", incluir="nao")],
    )
    antes = (corpus / "corpus_casos.csv").read_bytes()
    provider = FakeProvider(_resposta(_risco("Falha na pesquisa de precos", 3, causa="Pressa")))
    leitor = LeitorFalso()

    falhas, _ = _executar(corpus, provider, leitor)

    assert falhas == 0
    assert leitor.chamadas == [("AAA-01.pdf", 3, 4)]
    [prompt] = provider.calls
    assert "===== PÁGINA 3 =====" in prompt and "===== PÁGINA 4 =====" in prompt and "(pagina 4)" in prompt
    assert "===== PÁGINA 2 =====" not in prompt and "===== PÁGINA 5 =====" not in prompt
    assert "Usar so a secao 2" in prompt and "AAA-01" in prompt
    assert "<<" not in prompt
    assert (corpus / "corpus_casos.csv").read_bytes() == antes

    bruto = (corpus / "corpus_riscos_reais.csv").read_bytes()
    assert bruto.startswith(b"\xef\xbb\xbf" + ";".join(retro.RISCOS_FIELDS).encode())
    [linha] = _saida(corpus, "corpus_riscos_reais.csv")
    assert linha == {
        "processo_id": "AAA-01",
        "risco_id": "R01",
        "fase": "",
        "risco": "Falha na pesquisa de precos",
        "causa": "Pressa",
        "consequencia": "",
        "probabilidade": "",
        "impacto": "",
        "pagina": "3",
        "conferir": "nao",
    }
    [log] = _saida(corpus, "extracao_log.csv")
    assert (log["processo_id"], log["modelo"], log["versao_prompt"]) == ("AAA-01", "modelo-fake", "v1")
    assert (log["paginas_lidas"], log["qtd_riscos"], log["qtd_conferir"]) == ("3 a 4", "1", "0")
    assert log["data_hora"]


def test_risco_fora_do_texto_fica_com_conferir_sim(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01")])
    provider = FakeProvider(_resposta(_risco("Falha na pesquisa de precos"), _risco("Risco resumido pelo modelo")))
    _executar(corpus, provider, LeitorFalso())
    assert [l["conferir"] for l in _saida(corpus, "corpus_riscos_reais.csv")] == ["nao", "sim"]
    assert _saida(corpus, "extracao_log.csv")[0]["qtd_conferir"] == "1"


def test_retomada_nao_repete_processo_ja_extraido(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01")])
    provider = FakeProvider(_resposta(_risco("Falha na pesquisa de precos")))
    _executar(corpus, provider, LeitorFalso(), processos=["AAA-01"])
    assert len(provider.calls) == 1

    leitor = LeitorFalso()
    _executar(corpus, provider, leitor)
    assert [nome for nome, _, _ in leitor.chamadas] == ["BBB-01.pdf"]
    conteudo = (corpus / "corpus_riscos_reais.csv").read_bytes()

    leitor = LeitorFalso()
    _executar(corpus, provider, leitor)
    assert leitor.chamadas == [] and len(provider.calls) == 2
    assert (corpus / "corpus_riscos_reais.csv").read_bytes() == conteudo
    assert conteudo.count(b"\xef\xbb\xbf") == 1
    assert [l["processo_id"] for l in _saida(corpus, "corpus_riscos_reais.csv")] == ["AAA-01", "BBB-01"]


def test_falha_em_um_processo_nao_grava_nada_dele_e_e_retomada_depois(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01")])
    respostas = iter(["{ json cortado", _resposta(_risco("Falha na pesquisa de precos"))])
    falhas, mensagens = _executar(corpus, FakeProvider(lambda _prompt: next(respostas)), LeitorFalso())
    assert falhas == 1 and any("[FALHA] AAA-01" in m for m in mensagens)
    assert [l["processo_id"] for l in _saida(corpus, "extracao_log.csv")] == ["BBB-01"]

    leitor = LeitorFalso()
    falhas, _ = _executar(corpus, FakeProvider(_resposta(_risco("Falha na pesquisa de precos"))), leitor)
    assert falhas == 0 and [nome for nome, _, _ in leitor.chamadas] == ["AAA-01.pdf"]


def test_processo_sem_riscos_fica_registrado_no_log(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01")])
    provider = FakeProvider('{"riscos": []}')
    _, mensagens = _executar(corpus, provider, LeitorFalso())
    assert _saida(corpus, "extracao_log.csv")[0]["qtd_riscos"] == "0"
    assert any("nenhum risco" in m for m in mensagens)
    _executar(corpus, provider, LeitorFalso())
    assert len(provider.calls) == 1


def test_refazer_substitui_so_o_processo_pedido(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01")])
    _executar(corpus, FakeProvider(_resposta(_risco("Falha na pesquisa de precos"))), LeitorFalso())

    nova = _resposta(_risco("Dano: atraso"), {"risco_id": "R02", "risco": "Falha na pesquisa", "pagina": 2})
    leitor = LeitorFalso()
    _executar(corpus, FakeProvider(nova), leitor, refazer=["AAA-01"])

    assert [nome for nome, _, _ in leitor.chamadas] == ["AAA-01.pdf"]
    linhas = _saida(corpus, "corpus_riscos_reais.csv")
    assert [(l["processo_id"], l["risco"]) for l in linhas] == [
        ("BBB-01", "Falha na pesquisa de precos"),
        ("AAA-01", "Dano: atraso"),
        ("AAA-01", "Falha na pesquisa"),
    ]
    assert [l["processo_id"] for l in _saida(corpus, "extracao_log.csv")] == ["AAA-01", "BBB-01", "AAA-01"]


def test_processo_desconhecido_ou_excluido_e_erro(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01", incluir="nao")])
    for kwargs in ({"processos": ["ZZZ-01"]}, {"refazer": ["BBB-01"]}):
        with pytest.raises(retro.ExtracaoError):
            _executar(corpus, FakeProvider(_resposta()), LeitorFalso(), **kwargs)


def test_pdf_ausente_e_erro_antes_de_chamar_o_modelo(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01")])
    (corpus / "pdfs" / "AAA-01.pdf").unlink()
    provider = FakeProvider(_resposta())
    with pytest.raises(retro.ExtracaoError, match="AAA-01"):
        _executar(corpus, provider, LeitorFalso())
    assert provider.calls == []


# ---------------------------------------------------------------- leitura real do PDF


def _pdf(textos):
    """PDF minimo com camada de texto, uma pagina por item de ``textos``."""
    objetos = [b"<< /Type /Catalog /Pages 2 0 R >>", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    filhos = []
    for texto in textos:
        conteudo = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode("latin-1")
        numero_pagina = len(objetos) + 1
        filhos.append(f"{numero_pagina} 0 R")
        objetos.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {numero_pagina + 1} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objetos.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(conteudo), conteudo))
    objetos[1] = f"<< /Type /Pages /Kids [{' '.join(filhos)}] /Count {len(filhos)} >>".encode()

    saida = bytearray(b"%PDF-1.4\n")
    posicoes = []
    for numero, corpo in enumerate(objetos, start=1):
        posicoes.append(len(saida))
        saida += b"%d 0 obj\n%s\nendobj\n" % (numero, corpo)
    inicio_xref = len(saida)
    saida += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objetos) + 1)
    for posicao in posicoes:
        saida += b"%010d 00000 n \n" % posicao
    saida += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objetos) + 1, inicio_xref)
    return bytes(saida)


def test_leitura_real_respeita_o_intervalo_de_paginas(tmp_path):
    pytest.importorskip("pdfplumber")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(_pdf(["Estudo preliminar", "Risco 01: Falha na pesquisa", "Risco 02: Licitacao deserta", "Termo"]))

    paginas = retro.extrair_paginas(pdf, 2, 3)

    assert [p.numero for p in paginas] == [2, 3]
    assert "Falha na pesquisa" in paginas[0].texto and "Licitacao deserta" in paginas[1].texto
    assert "Licitacao deserta" in paginas[1].layout
    assert [w.texto for w in paginas[0].palavras][:2] == ["Risco", "01:"]
    documento = retro.montar_documento(paginas)
    assert "Estudo preliminar" not in documento and "Termo" not in documento

    with pytest.raises(retro.ExtracaoError, match="tem 4"):
        retro.extrair_paginas(pdf, 3, 5)
