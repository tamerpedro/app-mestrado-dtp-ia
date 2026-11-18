"""Painel de juizes (``retro_julgar.py --painel`` e ``retro_painel.py``). Sem rede e sem modelo."""

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from src.ai_provider import FakeProvider

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
juizes = _carregar("retro_juizes")
painel = _carregar("retro_painel")

CASOS = "processo_id;grupo;orgao;ano;arquivo;paginas_mapa;objeto;tipo_sugerido;objeto_confirmado;incluir;analise_principal;observacao"
RISCOS = "processo_id;risco_id;fase;risco;causa;consequencia;probabilidade;impacto;pagina;conferir"
CANDIDATOS = "processo_id;candidato_id;origem;id_catalogo;categoria;titulo;causa;consequencia"
PROCESSOS = ("AAA-01", "BBB-01")
NOMES = ["anthropic", "openai", "google"]


def _gravar(caminho, cabecalho, linhas):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(("\r\n".join([cabecalho, *linhas]) + "\r\n").encode("utf-8-sig"))


def _ler(caminho):
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


def _marcar(prompt, regra):
    """Resposta JSON que marca os candidatos cujo titulo casa com ``regra(numero do risco)``."""
    numero = re.search(r"- Risco: Risco real (\d+) de", prompt).group(1)
    padrao = regra(numero)
    cobertos = re.findall(rf"\[(C\d+)\] Título: [^\n]*{padrao}\n", prompt) if padrao else []
    return json.dumps({"cobertos": cobertos, "justificativa": f"Sobre o risco {numero}."})


def _como_referencia(prompt):
    return _marcar(prompt, lambda n: f"cobre R{n}")


class JuizFalso:
    """Juiz sem rede: ``responder(prompt, numero da chamada)`` devolve o texto ou uma excecao."""

    modelo = "modelo-falso"
    parametros = "acesso=teste; temperature=0"

    def __init__(self, nome, responder=lambda prompt, _n: _como_referencia(prompt)):
        self.nome = nome
        self.responder = responder
        self.prompts = []

    def perguntar(self, prompt):
        self.prompts.append(prompt)
        resposta = self.responder(prompt, len(self.prompts))
        if isinstance(resposta, Exception):
            raise resposta
        return juizes.Resposta(resposta, 100, 10)


@pytest.fixture
def corpus(tmp_path):
    """2 processos, 3 riscos reais e 7 candidatos cada, ja julgados pela referencia (semente 7).

    Referencia: R1 -> um de catalogo; R2 -> um de IA; R3 -> um de catalogo e um de IA.
    """
    pasta = tmp_path / "corpus"
    _gravar(pasta / "corpus_casos.csv", CASOS, [f"{pid};externo;Órgão;2024;{pid}.pdf;1;Objeto;aquisicao;sim;sim;sim;" for pid in PROCESSOS])
    reais, candidatos = [], []
    for pid in PROCESSOS:
        reais += [f"{pid};R{n};planejamento;Risco real {n} de {pid};Causa real {n};Dano {n};Baixa;Alta;1;nao" for n in (1, 2, 3)]
        candidatos += [
            f"{pid};{pid}_cat_R001;catalogo;R001;planejamento;Catálogo cobre R1;Causa c1;Efeito c1",
            f"{pid};{pid}_cat_R002;catalogo;R002;selecao;Catálogo cobre R3;Causa c2;Efeito c2",
            f"{pid};{pid}_cat_R003;catalogo;R003;gestao;Catálogo sem par;Causa c3;Efeito c3",
            f"{pid};{pid}_cat_R004;catalogo;R004;gestao;Catálogo avulso;Causa c4;Efeito c4",
            f"{pid};{pid}_ia_planejamento_1;ia;;planejamento;Gerado cobre R2;Causa i1;Efeito i1",
            f"{pid};{pid}_ia_selecao_1;ia;;selecao;Gerado cobre R3;Causa i2;Efeito i2",
            f"{pid};{pid}_ia_gestao_1;ia;;gestao;Gerado avulso;Causa i3;Efeito i3",
        ]
    _gravar(pasta / "corpus_riscos_reais.csv", RISCOS, reais)
    _gravar(pasta / "retrospectiva" / "candidatos.csv", CANDIDATOS, candidatos)
    referencia = FakeProvider(_como_referencia)
    assert julgar.julgar_cobertura(pasta, referencia, modelo="juiz-de-referencia", semente=7, saida=lambda _m: None) == 0
    PROMPTS_DA_REFERENCIA[pasta] = list(referencia.calls)
    return pasta


PROMPTS_DA_REFERENCIA: dict[Path, list[str]] = {}


def _rodar(corpus, juiz, **kwargs):
    mensagens = []
    relogio = iter(f"2026-11-18T10:00:{n:02d}-03:00" for n in range(60))
    falharam = julgar.julgar_painel(corpus, juiz, saida=mensagens.append, agora=lambda: next(relogio), **kwargs)
    return falharam, mensagens


def _arquivo(corpus, nome, prefixo="julgamento"):
    return corpus / "retrospectiva" / "painel" / f"{prefixo}_{nome}.csv"


# ---------------------------------------------------------------- mesma entrada da referencia


def test_painel_recebe_exatamente_a_entrada_da_referencia(corpus):
    referencia = (corpus / "retrospectiva" / "julgamento.csv").read_bytes()
    juiz = JuizFalso("openai")
    assert _rodar(corpus, juiz)[0] == 0
    assert juiz.prompts == PROMPTS_DA_REFERENCIA[corpus]  # mesmo prompt, mesmos candidatos, mesma ordem
    assert len(juiz.prompts) == 6 and "[C07] Título:" in juiz.prompts[0]
    assert (corpus / "retrospectiva" / "julgamento.csv").read_bytes() == referencia  # a referencia nao muda

    # a ordem vem da semente gravada em julgamento.csv (7), nao da semente padrao
    assert julgar.semente_do_julgamento(corpus / "retrospectiva") == 7
    candidatos = julgar.ler_candidatos(corpus / "retrospectiva")["AAA-01"]
    titulos = lambda semente: [c["titulo"] for _, c in julgar.embaralhar(candidatos, semente, "AAA-01", "R1")]  # noqa: E731
    apresentados = re.findall(r"\[C\d+\] Título: ([^\n]*)\n", juiz.prompts[0])
    assert apresentados == titulos(7) != titulos(julgar.SEMENTE_PADRAO)


def test_painel_grava_as_colunas_da_referencia_mais_o_registro_da_chamada(corpus):
    juiz = JuizFalso("google")
    falharam, mensagens = _rodar(corpus, juiz)
    assert falharam == 0
    linhas = _ler(_arquivo(corpus, "google"))
    referencia = _ler(corpus / "retrospectiva" / "julgamento.csv")
    assert list(linhas[0]) == julgar.PAINEL_FIELDS and list(linhas[0])[: len(julgar.JULGAMENTO_FIELDS)] == list(referencia[0])
    assert [(l["processo_id"], l["risco_id"], l["candidatos_cobertos"], l["cobertura"]) for l in linhas] == [
        (l["processo_id"], l["risco_id"], l["candidatos_cobertos"], l["cobertura"]) for l in referencia
    ]
    primeira = linhas[0]
    assert primeira["modelo_juiz"] == "modelo-falso" and primeira["versao_prompt_juiz"] == "v1" and primeira["semente"] == "7"
    assert primeira["parametros"] == "acesso=teste; temperature=0"
    assert primeira["data_hora"].startswith("2026-11-18T10:00:") and primeira["tentativas"] == "1"
    assert (primeira["tokens_entrada"], primeira["tokens_saida"]) == ("100", "10")

    chamadas = _ler(_arquivo(corpus, "google", "chamadas"))
    assert list(chamadas[0]) == julgar.CHAMADAS_FIELDS and len(chamadas) == 6
    assert {c["situacao"] for c in chamadas} == {"ok"} and all(c["tokens_entrada"] == "100" for c in chamadas)
    # a saida do terminal nao traz o veredito (a conferencia do autor e as cegas)
    assert not any(palavra in m for m in mensagens if m.startswith("[ok]") for palavra in ("catalogo", "ambos", "nenhum", "ia"))


def test_painel_limite_e_dry_run(corpus):
    juiz = JuizFalso("anthropic")
    falharam, mensagens = _rodar(corpus, juiz, dry_run=True)
    assert falharam == 0 and juiz.prompts == [] and not _arquivo(corpus, "anthropic").exists()
    assert any("6 a julgar" in m for m in mensagens) and any("semente 7" in m for m in mensagens)

    assert _rodar(corpus, juiz, limite=1)[0] == 0
    assert [l["processo_id"] for l in _ler(_arquivo(corpus, "anthropic"))] == ["AAA-01"] * 3
    assert _rodar(corpus, juiz)[0] == 0  # retoma: so o segundo processo
    assert len(juiz.prompts) == 6 and len(_ler(_arquivo(corpus, "anthropic"))) == 6


# ---------------------------------------------------------------- falhas


def test_resposta_invalida_tem_uma_nova_tentativa(corpus):
    juiz = JuizFalso("openai", lambda prompt, n: "nao sei" if n == 1 else _como_referencia(prompt))
    assert _rodar(corpus, juiz)[0] == 0
    primeira = _ler(_arquivo(corpus, "openai"))[0]
    assert primeira["cobertura"] == "catalogo" and primeira["tentativas"] == "2"
    assert (primeira["tokens_entrada"], primeira["tokens_saida"]) == ("200", "20")  # as duas chamadas contam
    chamadas = _ler(_arquivo(corpus, "openai", "chamadas"))
    assert [(c["risco_id"], c["tentativa"], c["situacao"]) for c in chamadas[:3]] == [
        ("R1", "1", "resposta invalida"), ("R1", "2", "ok"), ("R2", "1", "ok"),
    ]  # fmt: skip
    assert "sem objeto JSON" in chamadas[0]["erro"] and juiz.prompts[0] == juiz.prompts[1]


def test_duas_falhas_viram_sem_julgamento_nunca_nenhum_e_sao_refeitas(corpus):
    def responder(prompt, _n):
        if "Risco real 2 de AAA-01" in prompt:
            return '{"cobertos": ["C99"], "justificativa": "x"}'
        if "Risco real 1 de BBB-01" in prompt:
            return juizes.JuizError("APITimeoutError: tempo esgotado")
        return _como_referencia(prompt)

    juiz = JuizFalso("google", responder)
    falharam, mensagens = _rodar(corpus, juiz)
    assert falharam == 2 and len(juiz.prompts) == 8  # 6 riscos + 2 novas tentativas
    linhas = {(l["processo_id"], l["risco_id"]): l for l in _ler(_arquivo(corpus, "google"))}
    for chave, motivo in ((("AAA-01", "R2"), "fora da lista: C99"), (("BBB-01", "R1"), "tempo esgotado")):
        assert linhas[chave]["cobertura"] == "sem julgamento" != "nenhum"
        assert linhas[chave]["candidatos_cobertos"] == "" and linhas[chave]["tentativas"] == "2"
        assert linhas[chave]["justificativa"].startswith("[FALHA]") and motivo in linhas[chave]["justificativa"]
    assert sum(1 for m in mensagens if m.startswith("[FALHA]")) == 2
    situacoes = [c["situacao"] for c in _ler(_arquivo(corpus, "google", "chamadas"))]
    assert situacoes.count("resposta invalida") == 2 and situacoes.count("erro do provedor") == 2

    # retomada: so os dois sem julgamento sao perguntados de novo, e a linha antiga some
    bom = JuizFalso("google")
    falharam, mensagens = _rodar(corpus, bom)
    assert falharam == 0 and len(bom.prompts) == 2
    assert any("2 a julgar (2 deles sem julgamento" in m for m in mensagens)
    linhas = _ler(_arquivo(corpus, "google"))
    assert len(linhas) == 6 == len({(l["processo_id"], l["risco_id"]) for l in linhas})
    assert all(l["cobertura"] != "sem julgamento" for l in linhas)
    assert len(_ler(_arquivo(corpus, "google", "chamadas"))) == 10  # o historico das chamadas fica


def test_tres_falhas_seguidas_interrompem_e_configuracao_diferente_e_recusada(corpus):
    juiz = JuizFalso("anthropic", lambda _p, _n: juizes.JuizError("AuthenticationError: chave recusada"))
    falharam, mensagens = _rodar(corpus, juiz)
    assert falharam == 6 and len(juiz.prompts) == 6  # 3 riscos x 2 tentativas; os outros 3 nem foram tentados
    assert any("3 falhas seguidas" in m for m in mensagens)
    linhas = _ler(_arquivo(corpus, "anthropic"))
    assert [l["cobertura"] for l in linhas] == ["sem julgamento"] * 3

    outro = JuizFalso("anthropic")
    outro.parametros = "acesso=teste; temperature=1"
    with pytest.raises(julgar.JulgamentoError, match="outra configuracao"):
        _rodar(corpus, outro)


def test_juiz_nao_pode_ser_o_modelo_gerador(corpus):
    _gravar(
        corpus / "retrospectiva" / "geracao_log.csv",
        "processo_id;modelo;situacao",
        ["AAA-01;modelo-falso;gerado"],
    )
    with pytest.raises(julgar.JulgamentoError, match="mesmo que gerou"):
        _rodar(corpus, JuizFalso("openai"))


# ---------------------------------------------------------------- juizes e chaves


def test_juizes_do_painel_e_leitura_das_chaves(tmp_path):
    assert {nome: classe.modelo for nome, classe in juizes.JUIZES.items()} == {
        "anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna", "google": "gemini-3.1-flash-lite",
    }  # fmt: skip
    assert "Gemini API" in juizes.JuizGoogle.parametros and "temperature=0" in juizes.JuizGoogle.parametros
    assert "vertex" not in juizes.JuizGoogle.parametros.lower()
    assert all(f"={juizes.MAX_SAIDA}" in classe.parametros for classe in juizes.JUIZES.values())

    arquivo = tmp_path / "chaves.env"
    arquivo.write_text('# comentario\nOPENAI_API_KEY="sk-segredo-1"\nexport GEMINI_API_KEY = segredo-2\n\nlinha solta\n', encoding="utf-8")
    assert juizes.ler_chaves(arquivo) == {"OPENAI_API_KEY": "sk-segredo-1", "GEMINI_API_KEY": "segredo-2"}
    assert isinstance(juizes.construir("google", arquivo), juizes.JuizGoogle)
    with pytest.raises(juizes.JuizError, match="nao encontrado"):
        juizes.ler_chaves(tmp_path / "nao_existe.env")
    with pytest.raises(juizes.JuizError, match="Juiz desconhecido"):
        juizes.construir("outro", arquivo)
    with pytest.raises(juizes.JuizError, match="OPENAI_API_KEY nao configurada"):
        juizes.construir("openai", None).perguntar("x")


def test_erro_do_juiz_nunca_mostra_a_chave():
    class Quebrado(juizes.Juiz):
        nome, modelo, variavel = "quebrado", "m", "X_API_KEY"

        def _chamar(self, prompt):
            raise RuntimeError(f"401 chave invalida: {self._chave}")

    with pytest.raises(juizes.JuizError) as erro:
        Quebrado("sk-muito-secreta").perguntar("x")
    assert "sk-muito-secreta" not in str(erro.value) and "<chave>" in str(erro.value)
    assert erro.value.__cause__ is None


# ---------------------------------------------------------------- medidas de concordancia


def test_kappa_de_cohen():
    sim, nao = True, False
    assert painel.kappa_cohen([sim, sim, nao, nao], [sim, nao, nao, nao]) == pytest.approx(0.5)
    assert painel.kappa_cohen([sim, nao, sim, nao], [sim, nao, sim, nao]) == pytest.approx(1.0)
    assert painel.kappa_cohen([sim, nao], [nao, sim]) == pytest.approx(-1.0)
    assert painel.kappa_cohen([sim, sim, nao, nao], [sim, nao, sim, nao]) == pytest.approx(0.0)
    assert painel.kappa_cohen([nao, nao], [nao, nao]) is None  # sem variacao
    assert painel.kappa_cohen([], []) is None


def test_kappa_de_fleiss():
    sim, nao = True, False
    assert painel.kappa_fleiss([(sim, sim, sim), (nao, nao, nao), (sim, sim, nao), (nao, nao, sim)]) == pytest.approx(1 / 3)
    assert painel.kappa_fleiss([(sim, sim, sim), (nao, nao, nao)]) == pytest.approx(1.0)
    # com dois avaliadores e as mesmas proporcoes marginais, coincide com o de Cohen
    a, b = [sim, sim, nao, nao, sim, nao], [sim, nao, sim, nao, sim, nao]
    assert painel.kappa_fleiss(list(zip(a, b))) == pytest.approx(painel.kappa_cohen(a, b))
    assert painel.kappa_fleiss([(nao, nao, nao)]) is None and painel.kappa_fleiss([]) is None


def test_voto_da_maioria_exige_dois_dos_tres():
    assert painel.voto_da_maioria([{"a", "b"}, {"b", "c"}, {"c", "b"}]) == {"b", "c"}
    assert painel.voto_da_maioria([{"a"}, {"b"}, {"c"}]) == set()
    assert painel.voto_da_maioria([{"a"}, set(), set()]) == set()
    # juiz sem julgamento nao vota: os dois restantes precisam concordar
    assert painel.voto_da_maioria([{"a", "b"}, None, {"a"}]) == {"a"}
    assert painel.voto_da_maioria([{"a"}, None, None]) is None
    assert painel.voto_da_maioria([None, None, None]) is None


def _painel_de_tres(corpus):
    """anthropic = referencia; openai nao marca nada no R3; google so marca o 'sem par' e falha em AAA-01/R1."""
    _rodar(corpus, JuizFalso("anthropic"))
    _rodar(corpus, JuizFalso("openai", lambda p, _n: _marcar(p, lambda n: None if n == "3" else f"cobre R{n}")))

    def google(prompt, _n):
        return "sem json" if "Risco real 1 de AAA-01" in prompt else _marcar(prompt, lambda _numero: "sem par")

    _rodar(corpus, JuizFalso("google", google))


def test_analise_do_painel_contra_a_referencia(corpus):
    _painel_de_tres(corpus)
    r = painel.analisar(corpus)
    assert r["nomes"] == NOMES and len(r["escopo"]) == 6 == r["total_de_riscos"]

    # por juiz: classes e candidatos marcados; "sem julgamento" nao se mistura com "nenhum"
    assert dict(r["por_juiz"]["opus"]["classes"]) == {"catalogo": 2, "ia": 2, "ambos": 2}
    assert r["por_juiz"]["opus"]["marcados"] == 8 == r["por_juiz"]["anthropic"]["marcados"]
    assert dict(r["por_juiz"]["openai"]["classes"]) == {"catalogo": 2, "ia": 2, "nenhum": 2}
    assert dict(r["por_juiz"]["google"]["classes"]) == {"catalogo": 5, "sem julgamento": 1}
    assert r["por_juiz"]["google"]["marcados"] == 5 and r["por_juiz"]["google"]["riscos_julgados"] == 5
    # maioria: R1 e R2 ficam como na referencia; no R3 so um juiz marca cada candidato
    assert dict(r["por_juiz"]["maioria"]["classes"]) == {"catalogo": 2, "ia": 2, "nenhum": 2}
    assert r["votos"]["maioria"][("AAA-01", "R1")] == {"AAA-01_cat_R001"}  # com 2 juizes validos, os dois concordam
    assert r["votos"]["maioria"][("BBB-01", "R3")] == set()

    comparacoes = r["comparacoes"]
    assert comparacoes[("anthropic", "opus")] == {"pares": 42, "concordancia": 1.0, "kappa": pytest.approx(1.0)}
    assert comparacoes[("google", "opus")]["pares"] == 35  # o risco sem julgamento fica fora
    assert comparacoes[("openai", "google")]["pares"] == 35 and ("anthropic", "openai") in comparacoes
    # maioria x referencia: 42 pares, 8 marcados pela referencia, 4 pela maioria (todos entre os 8)
    geral = comparacoes[("maioria", "opus")]
    assert geral["pares"] == 42 and geral["concordancia"] == pytest.approx(38 / 42)
    esperado = (8 * 4 + 34 * 38) / 42**2
    assert geral["kappa"] == pytest.approx((38 / 42 - esperado) / (1 - esperado))
    assert r["fleiss"]["pares"] == 35 and r["fleiss"]["riscos"] == 5 and r["fleiss"]["kappa"] < geral["kappa"]

    assert r["maioria_x_referencia"] == {"riscos": 6, "sem_maioria": 0, "mesma_classe": 4, "mesmo_coberto": 4}
    assert len(r["divergencias"]) == 4
    assert {(d["risco_id"], d["opus"], d["maioria"]) for d in r["divergencias"]} == {("R3", True, False)}
    primeira = r["divergencias"][0]
    assert (primeira["anthropic"], primeira["openai"], primeira["google"]) == (True, False, False)

    # custos: 100 tokens de entrada e 10 de saida por chamada; google fez 7 chamadas (uma nova tentativa)
    custos = r["custos"]
    assert custos["anthropic"]["chamadas"] == 6 and custos["google"]["chamadas"] == 7 and custos["google"]["riscos"] == 6
    assert custos["google"]["situacoes"]["resposta invalida"] == 2
    assert custos["anthropic"]["custo"] == pytest.approx((600 * 0.10 + 60 * 0.50) / 1e6)
    assert custos["google"]["custo"] == pytest.approx((700 * 0.25 + 70 * 1.50) / 1e6)


def test_risco_com_um_so_juiz_valido_fica_sem_maioria(corpus):
    falha = lambda prompt, _n: "x" if "Risco real 1 de AAA-01" in prompt else _como_referencia(prompt)  # noqa: E731
    _rodar(corpus, JuizFalso("anthropic"))
    _rodar(corpus, JuizFalso("openai", falha))
    _rodar(corpus, JuizFalso("google", falha), limite=1)  # o segundo processo nem foi tentado pelo google
    r = painel.analisar(corpus)
    assert len(r["escopo"]) == 6
    assert r["por_juiz"]["google"]["classes"]["sem julgamento"] == 4  # 1 falha + 3 nao tentados
    assert r["votos"]["maioria"][("AAA-01", "R1")] is None
    assert r["maioria_x_referencia"]["riscos"] == 5 and r["maioria_x_referencia"]["sem_maioria"] == 1
    assert r["comparacoes"][("maioria", "opus")]["pares"] == 35
    assert r["maioria_x_referencia"]["mesmo_coberto"] == 5  # com 2 juizes validos, os dois concordam com a referencia


def test_relatorio_exemplos_e_arquivo_de_votos(corpus, capsys):
    _painel_de_tres(corpus)
    referencia = (corpus / "retrospectiva" / "julgamento.csv").read_bytes()
    # um dos dois riscos divergentes esta na amostra de conferencia do autor: nao pode virar exemplo
    _gravar(corpus / "retrospectiva" / "amostra_conferencia.csv", "processo_id;risco_id;risco_real", ["AAA-01;R3;texto"])
    assert painel.main(["--corpus", str(corpus), "--exemplos", "5"]) == 0
    saida = capsys.readouterr().out
    assert "Kappa de Fleiss dos 3 juizes" in saida and "Mesmo resultado coberto/nao coberto: 4/6 (66,7%)" in saida
    assert "claude-haiku-5-5" in saida and "gemini-3.1-flash-lite" in saida and "proj. 6 US$" in saida
    assert "2 exemplo(s) de par" in saida and "BBB-01 / R3" in saida and "AAA-01 / R3" not in saida
    assert "opus: cobre; anthropic: cobre; openai: nao cobre; google: nao cobre; maioria: nao cobre" in saida

    votos = _ler(corpus / "retrospectiva" / "painel" / "painel_votos.csv")
    assert list(votos[0]) == ["processo_id", "risco_id", "candidato_id", "opus", *NOMES, "maioria"]
    linha = next(v for v in votos if (v["processo_id"], v["risco_id"], v["candidato_id"]) == ("AAA-01", "R1", "AAA-01_cat_R001"))
    assert (linha["opus"], linha["anthropic"], linha["openai"], linha["google"], linha["maioria"]) == (
        "sim", "sim", "sim", "sem julgamento", "sim",
    )  # fmt: skip
    assert (corpus / "retrospectiva" / "julgamento.csv").read_bytes() == referencia

    assert painel.main(["--corpus", str(corpus), "--exemplos", "5", "--incluir-amostra"]) == 0
    assert "4 exemplo(s) de par" in capsys.readouterr().out


def test_relatorio_separa_a_analise_principal(corpus, capsys):
    _painel_de_tres(corpus)
    _gravar(
        corpus / "corpus_casos.csv",
        CASOS,
        [f"{pid};externo;Órgão;2024;{pid}.pdf;1;Objeto;aquisicao;sim;sim;{principal};" for pid, principal in zip(PROCESSOS, ("sim", "não"))],
    )
    assert painel.processos_da_analise_principal(corpus) == {"AAA-01"}
    r = painel.analisar(corpus, processos={"AAA-01"})
    assert len(r["escopo"]) == 3 == r["total_de_riscos"] and r["processos"] == 1
    assert dict(r["por_juiz"]["opus"]["classes"]) == {"catalogo": 1, "ia": 1, "ambos": 1}
    assert r["comparacoes"][("anthropic", "opus")]["pares"] == 21 and r["comparacoes"][("google", "opus")]["pares"] == 14
    assert r["maioria_x_referencia"] == {"riscos": 3, "sem_maioria": 0, "mesma_classe": 2, "mesmo_coberto": 2}
    assert r["custos"]["anthropic"]["chamadas"] == 3 and r["custos"]["google"]["chamadas"] == 4

    assert painel.main(["--corpus", str(corpus), "--exemplos", "0"]) == 0
    saida = capsys.readouterr().out
    principal, todos = saida.split("######## TODOS OS PROCESSOS INCLUIDOS ########")
    assert "ANALISE PRINCIPAL" in principal and "Escopo: 3 risco(s) reais julgados pelo painel, em 1 processo(s)" in principal
    assert "Escopo: 6 risco(s) reais julgados pelo painel, em 2 processo(s)" in todos
    assert "Mesmo resultado coberto/nao coberto: 2/3 (66,7%)" in principal
    assert "coberto" in todos and "4 (66,7%)" in todos  # maioria: 4 de 6 riscos cobertos


def test_painel_incompleto_e_recusado(corpus):
    _rodar(corpus, JuizFalso("anthropic"))
    with pytest.raises(painel.PainelError, match="openai, google"):
        painel.analisar(corpus)
