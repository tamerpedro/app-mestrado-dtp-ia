import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from src import ai_provider
from src.ai_provider import FakeProvider
from src.domain import CATEGORIES

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
julgar = _carregar("retro_julgar")
amostra = _carregar("retro_amostra")

CASOS = "processo_id;grupo;orgao;ano;arquivo;paginas_mapa;objeto;tipo_sugerido;objeto_confirmado;incluir;analise_principal;observacao"
RISCOS = "processo_id;risco_id;fase;risco;causa;consequencia;probabilidade;impacto;pagina;conferir"
CANDIDATOS = "processo_id;candidato_id;origem;id_catalogo;categoria;titulo;causa;consequencia"
CATEGORIAS_IA = ["planejamento", "selecao", "gestao"]


def _gravar(caminho, cabecalho, linhas, *, encoding="utf-8-sig", sep=";"):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    texto = "\r\n".join([cabecalho, *linhas]).replace(";", sep) + "\r\n"
    caminho.write_bytes(texto.encode(encoding))


def _corpus(tmp_path, processos=("AAA-01", "BBB-01"), riscos_por_processo=3, *, excluidos=(), **formato):
    """Cada processo: N riscos reais, 4 candidatos de catalogo e 3 de IA.

    Os titulos dos candidatos dizem o que o juiz falso deve responder:
    ``cobre R1`` = cobre o risco real R1 daquele processo.
    """
    corpus = tmp_path / "corpus"
    casos = [f"{pid};externo;Órgão;2024;{pid}.pdf;1;Aquisição de certificados SSL A1;aquisicao;sim;sim;sim;" for pid in processos]
    casos += [f"{pid};externo;Órgão;2024;{pid}.pdf;1;Objeto;aquisicao;sim;nao;sim;" for pid in excluidos]
    _gravar(corpus / "corpus_casos.csv", CASOS, casos, **formato)
    reais, candidatos = [], []
    for pid in (*processos, *excluidos):
        for n in range(1, riscos_por_processo + 1):
            reais.append(f"{pid};R{n};planejamento;Risco real {n} de {pid};Causa real {n};Dano {n};Baixa;Alta;1;nao")
        candidatos += [
            f"{pid};{pid}_cat_R001;catalogo;R001;planejamento;Catálogo cobre R1;Causa c1;Efeito c1",
            f"{pid};{pid}_cat_R002;catalogo;R002;selecao;Catálogo cobre R3;Causa c2;Efeito c2",
            f"{pid};{pid}_cat_R003;catalogo;R003;gestao;Catálogo sem par;Causa c3;Efeito c3",
            f"{pid};{pid}_cat_R004;catalogo;R004;gestao;Catálogo sem par também;Causa c4;Efeito c4",
        ]
        candidatos += [
            f"{pid};{pid}_ia_{categoria}_1;ia;;{categoria};{titulo};Causa i{i};Efeito i{i}"
            for i, (categoria, titulo) in enumerate(
                zip(CATEGORIAS_IA, ["Gerado cobre R2", "Gerado cobre R3", "Gerado sem par"]), start=1
            )
        ]
    _gravar(corpus / "corpus_riscos_reais.csv", RISCOS, reais)
    _gravar(corpus / "retrospectiva" / "candidatos.csv", CANDIDATOS, candidatos)
    return corpus


def _juiz_cobertura(prompt):
    """Marca os candidatos cujo titulo diz ``cobre R<n>`` para o risco real n."""
    numero = re.search(r"- Risco: Risco real (\d+) de", prompt).group(1)
    cobertos = re.findall(rf"\[(C\d+)\] Título: [^\n]*cobre R{numero}\n", prompt)
    return json.dumps({"cobertos": cobertos, "justificativa": f"Mesmo evento do   risco {numero}."})


def _juiz_riscos_ia(prompt):
    codigos = re.findall(r"\[(R\d+)\] Categoria:", prompt)
    respostas = [
        {"nomeia_objeto": "Sim", "termo": "gerado  cobre | Causa i1", "especifico_do_objeto": "sim", "grupo": 1},
        {"nomeia_objeto": "sim", "termo": "certificado SSL", "especifico_do_objeto": "não", "grupo": 1},  # nao esta no texto
        {"nomeia_objeto": "não", "termo": "", "especifico_do_objeto": "nao", "grupo": 2},
    ]
    itens = [{"id": c, **respostas[i]} for i, c in enumerate(codigos)]
    return "```json\n" + json.dumps({"riscos": itens}, ensure_ascii=False) + "\n```"


def _ler(caminho):
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


def _cobertura(corpus, provider, **kwargs):
    mensagens = []
    falharam = julgar.julgar_cobertura(corpus, provider, modelo="juiz-fake", saida=mensagens.append, **kwargs)
    return falharam, mensagens


def _riscos_ia(corpus, provider, **kwargs):
    mensagens = []
    falharam = julgar.julgar_riscos_ia(corpus, provider, modelo="juiz-fake", saida=mensagens.append, **kwargs)
    return falharam, mensagens


# ---------------------------------------------------------------- parsing das respostas


def test_interpreta_cobertura_com_texto_em_volta_e_codigos_em_qualquer_forma():
    codigos = ["C01", "C02", "C03"]
    texto = 'Segue:\n```json\n{"cobertos": ["c03", "[C01]", "C03"], "justificativa": " Mesmo\n evento. "}\n```'
    assert julgar.interpretar_cobertura(texto, codigos) == (["C01", "C03"], "Mesmo evento.")
    assert julgar.interpretar_cobertura('{"cobertos": [], "justificativa": "Nenhum."}', codigos) == ([], "Nenhum.")


@pytest.mark.parametrize(
    ("texto", "mensagem"),
    [
        ("nao sei", "sem objeto JSON"),
        ('{"cobertos": ["C01",}', "JSON invalido"),
        ('{"justificativa": "x"}', "'cobertos' ausente"),
        ('{"cobertos": "C01", "justificativa": "x"}', "'cobertos' ausente"),
        ('{"cobertos": ["C09"], "justificativa": "x"}', "fora da lista: C09"),
        ('{"cobertos": ["C01"]}', "'justificativa' ausente"),
    ],
)
def test_resposta_de_cobertura_invalida(texto, mensagem):
    with pytest.raises(julgar.RespostaInvalida, match=re.escape(mensagem)):
        julgar.interpretar_cobertura(texto, ["C01", "C02"])


def _item(**mudancas):
    return {"id": "R01", "nomeia_objeto": "sim", "termo": "SSL", "especifico_do_objeto": "nao", "grupo": 1, **mudancas}


def test_interpreta_riscos_ia():
    itens = [
        {"id": "R02", "nomeia_objeto": "NÃO", "termo": "", "especifico_do_objeto": "Sim", "grupo": "2"},
        {"id": "r01", "nomeia_objeto": "sim", "termo": ' "certificados  SSL" | chaves privadas ', "especifico_do_objeto": "não", "grupo": 1},
        {"id": "R03", "nomeia_objeto": "sim", "termo": ["navegadores"], "especifico_do_objeto": "sim", "grupo": 3},
        {"id": "R04", "nomeia_objeto": "nao", "especifico_do_objeto": "nao", "grupo": 3},
    ]
    assert julgar.interpretar_riscos_ia(json.dumps({"riscos": itens}), ["R01", "R02", "R03", "R04"]) == {
        "R01": {"nomeia_objeto": "sim", "termos": ["certificados SSL", "chaves privadas"], "especifico_do_objeto": "nao", "grupo": 1},
        "R02": {"nomeia_objeto": "nao", "termos": [], "especifico_do_objeto": "sim", "grupo": 2},
        "R03": {"nomeia_objeto": "sim", "termos": ["navegadores"], "especifico_do_objeto": "sim", "grupo": 3},
        "R04": {"nomeia_objeto": "nao", "termos": [], "especifico_do_objeto": "nao", "grupo": 3},
    }


@pytest.mark.parametrize(
    ("itens", "mensagem"),
    [
        ([_item()], "sem resposta: R02"),
        ([_item()] * 2, "id repetido: R01"),
        ([_item(id="R07")], "id fora da lista: R07"),
        ([_item(nomeia_objeto="talvez")], "nomeia_objeto invalido"),
        ([_item(especifico_do_objeto="")], "especifico_do_objeto invalido"),
        ([_item(termo=3)], "termo invalido"),
        ([_item(grupo="a")], "grupo invalido"),
        ([_item(grupo=True)], "grupo invalido"),
        ([_item(grupo=None)], "grupo invalido"),
    ],
)
def test_resposta_de_riscos_ia_invalida(itens, mensagem):
    with pytest.raises(julgar.RespostaInvalida, match=re.escape(mensagem)):
        julgar.interpretar_riscos_ia(json.dumps({"riscos": itens}), ["R01", "R02"])


@pytest.mark.parametrize(
    ("nomeia", "termos", "conferir"),
    [
        ("sim", ["certificados ssl"], "nao"),  # sem diferenciar maiusculas
        ("sim", ["Certificados SSL", "navegadores"], "nao"),  # titulo e consequencia
        ("nao", [], "nao"),
        ("sim", ["certificado digital"], "termo nao encontrado no texto: certificado digital"),
        ("sim", ["Certificados SSL", "A1", "ICP"], "termo nao encontrado no texto: A1 | ICP"),
        ("sim", ["Certificados SSL por falha"], "termo nao encontrado no texto: Certificados SSL por falha"),  # junta campos
        ("sim", [], "nomeia_objeto=sim sem termo"),
        ("nao", ["navegadores"], "termo preenchido com nomeia_objeto=nao"),
        ("nao", ["tokens"], "termo nao encontrado no texto: tokens; termo preenchido com nomeia_objeto=nao"),
    ],
)
def test_termo_tem_de_existir_literalmente_no_texto_do_risco(nomeia, termos, conferir):
    risco = {"titulo": "Expiração de Certificados SSL", "causa": "por falha de controle", "consequencia": "Erro nos navegadores"}
    assert julgar.conferir_termo(risco, nomeia, termos) == conferir


# ---------------------------------------------------------------- derivacao da cobertura


@pytest.mark.parametrize(
    ("origens", "cobertura"),
    [
        ([], "nenhum"),
        (["catalogo"], "catalogo"),
        (["catalogo", "catalogo"], "catalogo"),
        (["ia"], "ia"),
        (["ia", "catalogo", "ia"], "ambos"),
    ],
)
def test_derivar_cobertura(origens, cobertura):
    assert julgar.derivar_cobertura(origens) == cobertura


def test_julgamento_grava_cobertura_derivada_da_origem(tmp_path):
    corpus = _corpus(tmp_path, processos=("AAA-01",), riscos_por_processo=4)
    casos = (corpus / "corpus_casos.csv").read_bytes()
    provider = FakeProvider(_juiz_cobertura)

    falharam, _ = _cobertura(corpus, provider, semente=7)

    assert falharam == 0 and len(provider.calls) == 4
    assert (corpus / "corpus_casos.csv").read_bytes() == casos  # nunca regravado
    linhas = _ler(corpus / "retrospectiva" / "julgamento.csv")
    assert list(linhas[0]) == julgar.JULGAMENTO_FIELDS
    assert [(l["risco_id"], l["cobertura"]) for l in linhas] == [("R1", "catalogo"), ("R2", "ia"), ("R3", "ambos"), ("R4", "nenhum")]
    assert linhas[0]["candidatos_cobertos"] == "AAA-01_cat_R001"
    assert set(linhas[2]["candidatos_cobertos"].split(" | ")) == {"AAA-01_cat_R002", "AAA-01_ia_selecao_1"}
    assert linhas[3]["candidatos_cobertos"] == ""
    assert linhas[1]["justificativa"] == "Mesmo evento do risco 2."
    assert {(l["modelo_juiz"], l["versao_prompt_juiz"], l["semente"]) for l in linhas} == {("juiz-fake", "v1", "7")}


def test_justificativa_troca_os_codigos_pelos_ids_reais(tmp_path):
    por_codigo = {"C01": {"candidato_id": "AAA-01_cat_R001"}, "C12": {"candidato_id": "AAA-01_ia_gestao_1"}}
    texto = "C12 e C01 cobrem; C99 e o item C1 nao; (C01)."
    assert julgar.trocar_codigos(texto, por_codigo) == "AAA-01_ia_gestao_1 e AAA-01_cat_R001 cobrem; C99 e o item C1 nao; (AAA-01_cat_R001)."

    corpus = _corpus(tmp_path, processos=("AAA-01",), riscos_por_processo=3)

    def cita_os_codigos(prompt):
        resposta = json.loads(_juiz_cobertura(prompt))
        resposta["justificativa"] = f"Cobrem: {', '.join(resposta['cobertos'])}."
        return json.dumps(resposta)

    _cobertura(corpus, FakeProvider(cita_os_codigos))
    for linha in _ler(corpus / "retrospectiva" / "julgamento.csv"):
        assert linha["justificativa"] == f"Cobrem: {linha['candidatos_cobertos'].replace(' | ', ', ')}."
        assert not re.search(r"\bC\d\d\b", linha["justificativa"])


# ---------------------------------------------------------------- o que o juiz ve


def test_prompt_esconde_a_origem_e_embaralha_com_a_semente(tmp_path):
    corpus = _corpus(tmp_path, processos=("AAA-01",))
    provider = FakeProvider(_juiz_cobertura)
    _cobertura(corpus, provider, semente=7)

    for prompt in provider.calls:
        assert "Risco real" in prompt and "- Causa: Causa real" in prompt and "- Consequência: Dano" in prompt
        assert re.findall(r"\[(C\d+)\]", prompt) == [f"C{n:02d}" for n in range(1, 8)]
        for vazamento in ("AAA-01_", "_cat_", "_ia_", "catalogo", "R001", "planejamento"):
            assert vazamento not in prompt
        assert "{" not in prompt.split("## Formato de saída")[0]  # nenhum campo sem preencher

    ordem = [re.findall(r"Título: (.+)", prompt) for prompt in provider.calls]
    no_arquivo = [l["titulo"] for l in _ler(corpus / "retrospectiva" / "candidatos.csv")]
    assert all(sorted(o) == sorted(no_arquivo) for o in ordem)
    assert any(o != no_arquivo for o in ordem) and len({tuple(o) for o in ordem}) > 1  # embaralha, e por risco

    candidatos = julgar.ler_candidatos(corpus / "retrospectiva")["AAA-01"]
    mesma = julgar.embaralhar(candidatos, 7, "AAA-01", "R1")
    assert mesma == julgar.embaralhar(candidatos, 7, "AAA-01", "R1")
    assert [c["titulo"] for _, c in mesma] == ordem[0]
    assert mesma != julgar.embaralhar(candidatos, 8, "AAA-01", "R1")


def test_campo_vazio_do_risco_real_vai_como_nao_informado(tmp_path):
    corpus = _corpus(tmp_path, processos=("AAA-01",), riscos_por_processo=1)
    reais = (corpus / "corpus_riscos_reais.csv").read_text(encoding="utf-8-sig").replace("Causa real 1", "")
    (corpus / "corpus_riscos_reais.csv").write_text(reais, encoding="utf-8-sig")
    provider = FakeProvider(_juiz_cobertura)
    _cobertura(corpus, provider)
    assert "- Causa: (não informada)\n" in provider.calls[0]


# ---------------------------------------------------------------- retomada, limite e falhas


def test_retomada_pula_o_que_ja_foi_julgado(tmp_path):
    corpus = _corpus(tmp_path)
    chamadas = {"n": 0}

    def falha_no_terceiro(prompt):
        chamadas["n"] += 1
        return "sem json" if chamadas["n"] in (3, 4) else _juiz_cobertura(prompt)  # a nova tentativa tambem falha

    falharam, mensagens = _cobertura(corpus, FakeProvider(falha_no_terceiro))
    assert falharam == 1 and any("[FALHA] AAA-01 / R3" in m for m in mensagens)
    caminho = corpus / "retrospectiva" / "julgamento.csv"
    assert [(l["processo_id"], l["risco_id"]) for l in _ler(caminho)] == [
        ("AAA-01", "R1"), ("AAA-01", "R2"), ("BBB-01", "R1"), ("BBB-01", "R2"), ("BBB-01", "R3"),
    ]  # fmt: skip
    antes = caminho.read_bytes()

    provider = FakeProvider(_juiz_cobertura)
    falharam, mensagens = _cobertura(corpus, provider)
    assert falharam == 0 and len(provider.calls) == 1 and "Risco real 3 de AAA-01" in provider.calls[0]
    assert any("5 ja julgado(s); 1 a julgar" in m for m in mensagens)
    assert caminho.read_bytes().startswith(antes) and len(_ler(caminho)) == 6

    provider = FakeProvider(_juiz_cobertura)
    assert _cobertura(corpus, provider)[0] == 0 and provider.calls == []


def test_resposta_invalida_ganha_uma_nova_tentativa(tmp_path):
    corpus = _corpus(tmp_path, processos=("AAA-01",), riscos_por_processo=1)
    chamadas = {"n": 0}

    def primeira_invalida(prompt):
        chamadas["n"] += 1
        return "{}" if chamadas["n"] == 1 else _juiz_cobertura(prompt)

    provider = FakeProvider(primeira_invalida)
    assert _cobertura(corpus, provider)[0] == 0 and len(provider.calls) == 2
    assert len(_ler(corpus / "retrospectiva" / "julgamento.csv")) == 1


def test_nao_mistura_modelo_prompt_ou_semente_no_mesmo_arquivo(tmp_path):
    corpus = _corpus(tmp_path)
    _cobertura(corpus, FakeProvider(_juiz_cobertura), limite=1)
    provider = FakeProvider(_juiz_cobertura)
    with pytest.raises(julgar.JulgamentoError, match="semente=20261117"):
        _cobertura(corpus, provider, semente=99)
    with pytest.raises(julgar.JulgamentoError, match="modelo_juiz=juiz-fake"):
        julgar.julgar_cobertura(corpus, provider, modelo="outro-juiz", saida=lambda _m: None)
    assert provider.calls == []


def test_limite_julga_so_os_primeiros_processos_e_ignora_os_excluidos(tmp_path):
    corpus = _corpus(tmp_path, processos=("AAA-01", "BBB-01", "CCC-01"), excluidos=("XXX-01",), encoding="cp1252", sep="\t")
    provider = FakeProvider(_juiz_cobertura)
    _cobertura(corpus, provider, limite=2)
    assert {l["processo_id"] for l in _ler(corpus / "retrospectiva" / "julgamento.csv")} == {"AAA-01", "BBB-01"}
    _cobertura(corpus, provider)  # depois, sem limite: so o que falta; incluir=nao fica de fora
    assert len(provider.calls) == 9
    assert {l["processo_id"] for l in _ler(corpus / "retrospectiva" / "julgamento.csv")} == {"AAA-01", "BBB-01", "CCC-01"}


def test_dry_run_nao_chama_o_juiz_nem_grava(tmp_path):
    corpus = _corpus(tmp_path)
    provider = FakeProvider(_juiz_cobertura)
    falharam, mensagens = _cobertura(corpus, provider, dry_run=True)
    assert falharam == 0 and provider.calls == [] and any("6 a julgar" in m for m in mensagens)
    assert _riscos_ia(corpus, provider, dry_run=True)[0] == 0 and provider.calls == []
    assert sorted(p.name for p in (corpus / "retrospectiva").iterdir()) == ["candidatos.csv"]


def test_provedor_recusando_interrompe_a_execucao(tmp_path):
    corpus = _corpus(tmp_path)
    provider = FakeProvider(fail=True)
    falharam, mensagens = _cobertura(corpus, provider)
    assert falharam == 6 and len(provider.calls) == 3
    assert any("execucao interrompida (3 risco(s) nem foram tentados)" in m for m in mensagens)
    assert not (corpus / "retrospectiva" / "julgamento.csv").exists()


def test_recusa_corpus_no_repositorio_e_juiz_igual_ao_gerador(tmp_path):
    with pytest.raises(julgar.JulgamentoError, match="dentro do repositorio"):
        _cobertura(ROOT / "data", FakeProvider(_juiz_cobertura))
    corpus = _corpus(tmp_path)
    log = "processo_id;provedor;modelo;situacao"
    _gravar(corpus / "retrospectiva" / "geracao_log.csv", log, ["AAA-01;anthropic;juiz-fake;gerado"])
    provider = FakeProvider(_juiz_cobertura)
    with pytest.raises(julgar.JulgamentoError, match="mesmo que gerou"):
        _cobertura(corpus, provider)
    with pytest.raises(julgar.JulgamentoError, match="mesmo que gerou"):
        _riscos_ia(corpus, provider)
    assert provider.calls == []


def test_juiz_vem_de_judge_provider_e_judge_model(monkeypatch):
    for variavel in ("JUDGE_PROVIDER", "JUDGE_MODEL"):  # o desenvolvedor pode te-las definidas no terminal
        monkeypatch.delenv(variavel, raising=False)
    with pytest.raises(julgar.JulgamentoError, match="JUDGE_PROVIDER e JUDGE_MODEL"):
        julgar.construir_juiz()
    monkeypatch.setenv("AI_MODEL", "modelo-gerador")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "chave-de-teste")
    monkeypatch.setenv("JUDGE_PROVIDER", "Anthropic")
    monkeypatch.setenv("JUDGE_MODEL", "modelo-gerador")
    with pytest.raises(julgar.JulgamentoError, match="diferente do modelo gerador"):
        julgar.construir_juiz()
    monkeypatch.setenv("JUDGE_MODEL", "modelo-juiz")
    provider, provedor, modelo = julgar.construir_juiz(timeout=45)
    assert isinstance(provider, ai_provider.AnthropicProvider) and (provedor, modelo) == ("anthropic", "modelo-juiz")
    assert (provider.model, provider.timeout, provider.max_retries) == ("modelo-juiz", 45, 1)
    assert provider.max_tokens == julgar.MAX_TOKENS_JUIZ > 2048  # o raciocinio do modelo conta no limite
    assert ai_provider.TEMPERATURE == 0.0


# ---------------------------------------------------------------- modo --riscos-ia


def test_riscos_ia_grava_uma_linha_por_risco_de_ia(tmp_path):
    corpus = _corpus(tmp_path)
    provider = FakeProvider(_juiz_riscos_ia)

    falharam, mensagens = _riscos_ia(corpus, provider)

    assert falharam == 0 and len(provider.calls) == 2  # uma chamada por processo
    prompt = provider.calls[0]
    assert "- Objeto: Aquisição de certificados SSL A1" in prompt and "- Tipo de contratação: aquisicao" in prompt
    assert re.findall(r"\[(R\d+)\] Categoria: (\w+)", prompt) == [("R01", "planejamento"), ("R02", "selecao"), ("R03", "gestao")]
    assert "Catálogo" not in prompt and "Gerado cobre R2" in prompt  # so os riscos de IA
    linhas = _ler(corpus / "retrospectiva" / "riscos_ia.csv")
    assert list(linhas[0]) == julgar.RISCOS_IA_FIELDS
    assert list(linhas[0]) == [
        "processo_id", "candidato_id", "categoria", "nomeia_objeto", "termo", "especifico_do_objeto", "grupo",
        "modelo_juiz", "versao_prompt_juiz", "conferir",
    ]  # fmt: skip
    campos = ("candidato_id", "categoria", "nomeia_objeto", "termo", "especifico_do_objeto", "grupo", "conferir")
    assert [tuple(l[c] for c in campos) for l in linhas[:3]] == [
        ("AAA-01_ia_planejamento_1", "planejamento", "sim", "gerado cobre | Causa i1", "sim", "1", "nao"),
        ("AAA-01_ia_selecao_1", "selecao", "sim", "certificado SSL", "nao", "1", "termo nao encontrado no texto: certificado SSL"),
        ("AAA-01_ia_gestao_1", "gestao", "nao", "", "nao", "2", "nao"),
    ]
    assert {(l["modelo_juiz"], l["versao_prompt_juiz"]) for l in linhas} == {("juiz-fake", "v2")}
    assert "nomeia_objeto" in prompt and "especifico_do_objeto" in prompt and "{" not in prompt.split("## Formato de saída")[0]
    assert (ROOT / "prompts" / "juiz_riscos_ia_v1.txt").is_file()  # o v1 fica como registro do piloto
    assert any(
        "[ok] AAA-01: 3 risco(s) de IA; 2 nomeiam o objeto; 1 especificos; 2 grupo(s); 1 a conferir." in m for m in mensagens
    )
    assert not (corpus / "retrospectiva" / "julgamento.csv").exists()


def test_riscos_ia_e_retomavel_e_aceita_limite(tmp_path):
    corpus = _corpus(tmp_path, processos=("AAA-01", "BBB-01", "CCC-01"))
    chamadas = {"n": 0}

    def segundo_incompleto(prompt):
        chamadas["n"] += 1
        if chamadas["n"] in (2, 3):  # BBB-01, inclusive na nova tentativa
            return json.dumps({"riscos": [_item()]})
        return _juiz_riscos_ia(prompt)

    falharam, mensagens = _riscos_ia(corpus, FakeProvider(segundo_incompleto), limite=2)
    caminho = corpus / "retrospectiva" / "riscos_ia.csv"
    assert falharam == 1 and any("[FALHA] BBB-01: risco(s) sem resposta: R02, R03" in m for m in mensagens)
    assert {l["processo_id"] for l in _ler(caminho)} == {"AAA-01"}  # nada parcial de BBB-01

    provider = FakeProvider(_juiz_riscos_ia)
    falharam, mensagens = _riscos_ia(corpus, provider)
    assert falharam == 0 and len(provider.calls) == 2 and any("1 ja classificado(s); 2 a classificar" in m for m in mensagens)
    assert [l["processo_id"] for l in _ler(caminho)] == ["AAA-01"] * 3 + ["BBB-01"] * 3 + ["CCC-01"] * 3


# ---------------------------------------------------------------- amostra para conferencia


def _julgado(tmp_path, processos, riscos_por_processo):
    corpus = _corpus(tmp_path, processos=processos, riscos_por_processo=riscos_por_processo)
    _cobertura(corpus, FakeProvider(_juiz_cobertura))
    return corpus


def test_tamanho_da_amostra():
    assert amostra.tamanho_da_amostra(249, 23) == 50  # 20%
    assert amostra.tamanho_da_amostra(120, 10) == 40  # minimo
    assert amostra.tamanho_da_amostra(30, 5) == 30  # nunca mais que o total
    assert amostra.tamanho_da_amostra(300, 70) == 70  # um por processo


def test_amostra_sorteia_com_semente_e_um_risco_por_processo(tmp_path):
    processos = tuple(f"P{n:02d}-01" for n in range(1, 11))
    corpus = _julgado(tmp_path, processos, riscos_por_processo=25)  # 250 riscos -> 50 sorteados
    mensagens = []
    assert amostra.gerar_amostra(corpus, saida=mensagens.append) == 50

    linhas = _ler(corpus / "retrospectiva" / "amostra_conferencia.csv")
    assert list(linhas[0]) == amostra.AMOSTRA_FIELDS
    sorteados = list(dict.fromkeys((l["processo_id"], l["risco_id"]) for l in linhas))
    assert len(sorteados) == 50 and {pid for pid, _ in sorteados} == set(processos)
    assert len(linhas) == 50 * 7  # uma linha por candidato do processo
    assert all(l["cobre_autor"] == "" and l["observacao"] == "" for l in linhas)
    assert any("50 de 250 risco(s)" in m and "semente 20261117" in m for m in mensagens)

    julgados = [(pid, f"R{n}") for pid in processos for n in range(1, 26)]
    assert sorteados == amostra.sortear(julgados, amostra.SEMENTE_PADRAO)  # reprodutivel
    assert sorteados != amostra.sortear(julgados, 1)


def test_amostra_nao_traz_a_resposta_do_juiz_nem_a_origem(tmp_path):
    corpus = _julgado(tmp_path, ("AAA-01", "BBB-01"), riscos_por_processo=3)
    amostra.gerar_amostra(corpus, saida=lambda _m: None)

    bruto = (corpus / "retrospectiva" / "amostra_conferencia.csv").read_text(encoding="utf-8-sig")
    for vazamento in ("_cat_", "_ia_", "catalogo", "ambos", "nenhum", "juiz-fake", "Mesmo evento", "R001"):
        assert vazamento not in bruto
    linhas = _ler(corpus / "retrospectiva" / "amostra_conferencia.csv")
    assert len(linhas) == 6 * 7  # minimo de 40 > 6 riscos: entram todos
    assert linhas[0]["risco_real"] == "Risco real 1 de AAA-01 [Causa: Causa real 1] [Consequência: Dano 1]"

    # codigos neutros, embaralhados e iguais para todos os riscos do processo
    por_risco = {}
    for l in linhas:
        por_risco.setdefault((l["processo_id"], l["risco_id"]), []).append((l["candidato_id"], l["titulo_candidato"]))
    assert por_risco[("AAA-01", "R1")] == por_risco[("AAA-01", "R2")] == por_risco[("AAA-01", "R3")]
    assert [codigo for codigo, _ in por_risco[("AAA-01", "R1")]] == [f"K{n:02d}" for n in range(1, 8)]
    no_arquivo = [l["titulo"] for l in _ler(corpus / "retrospectiva" / "candidatos.csv") if l["processo_id"] == "AAA-01"]
    assert [titulo for _, titulo in por_risco[("AAA-01", "R1")]] != no_arquivo
    assert list(linhas[0])[5:7] == ["causa_candidato", "consequencia_candidato"]

    # a chave devolve o candidato original
    chave = _ler(corpus / "retrospectiva" / "amostra_chave.csv")
    assert list(chave[0]) == amostra.CHAVE_FIELDS and len(chave) == 2 * 7
    candidatos = {l["candidato_id"]: l for l in _ler(corpus / "retrospectiva" / "candidatos.csv")}
    for c in chave:
        linha = next(l for l in linhas if (l["processo_id"], l["candidato_id"]) == (c["processo_id"], c["candidato_codigo"]))
        original = candidatos[c["candidato_id"]]
        assert (linha["titulo_candidato"], linha["causa_candidato"]) == (original["titulo"], original["causa"])
        assert linha["consequencia_candidato"] == original["consequencia"]

    # ordenada por processo, risco e categoria do candidato
    categoria = {(c["processo_id"], c["candidato_codigo"]): candidatos[c["candidato_id"]]["categoria"] for c in chave}
    ordem = [
        (l["processo_id"], int(l["risco_id"][1:]), CATEGORIES.index(categoria[(l["processo_id"], l["candidato_id"])]))
        for l in linhas
    ]
    assert ordem == sorted(ordem)
    assert [categoria[("AAA-01", f"K{n:02d}")] for n in range(1, 8)] == [
        "planejamento", "planejamento", "selecao", "selecao", "gestao", "gestao", "gestao",
    ]  # fmt: skip


def test_amostra_nao_sobrescreve_e_exige_julgamento_completo(tmp_path):
    corpus = _corpus(tmp_path)
    _cobertura(corpus, FakeProvider(_juiz_cobertura), limite=1)
    with pytest.raises(amostra.AmostraError, match="So 3 de 6 riscos reais"):
        amostra.gerar_amostra(corpus, saida=lambda _m: None)
    assert amostra.gerar_amostra(corpus, permitir_parcial=True, saida=lambda _m: None) == 3

    caminho = corpus / "retrospectiva" / "amostra_conferencia.csv"
    marcado = caminho.read_bytes().replace(b";;\r\n", b";sim;\r\n", 1)
    caminho.write_bytes(marcado)
    with pytest.raises(amostra.AmostraError, match="ja existe"):
        amostra.gerar_amostra(corpus, permitir_parcial=True, saida=lambda _m: None)
    assert caminho.read_bytes() == marcado
