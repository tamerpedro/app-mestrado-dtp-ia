import csv
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from src.ai_provider import FakeProvider
from src.case_store import load_case
from src.domain import CATEGORIES
from src.risk_library import load_risks

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
gerar = _carregar("retro_gerar")

BIBLIOTECA = load_risks(gerar.BIBLIOTECA)
ESPERADO = gerar.quantidades_esperadas(BIBLIOTECA)
CHAMADAS = sum(1 for quantidade in ESPERADO.values() if quantidade)
TOTAL_IA = sum(ESPERADO.values())

CABECALHO = "processo_id;grupo;orgao;ano;arquivo;paginas_mapa;objeto;tipo_sugerido;objeto_confirmado;incluir;analise_principal;observacao"


def _linha(pid, objeto="Aquisição de certificados SSL A1", tipo="aquisicao", incluir="sim", principal="sim"):
    return f"{pid};externo;Órgão;2024;{pid}.pdf;1 a 2;{objeto};{tipo};sim;{incluir};{principal};"


def _corpus(tmp_path, linhas, *, encoding="utf-8-sig", sep=";", com_riscos=None):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    texto = "\r\n".join([CABECALHO, *linhas]).replace(";", sep) + "\r\n"
    (corpus / "corpus_casos.csv").write_bytes(texto.encode(encoding))
    ids = [linha.split(";")[0] for linha in linhas] if com_riscos is None else com_riscos
    riscos = ["processo_id;risco_id;fase;risco;causa;consequencia;probabilidade;impacto;pagina;conferir"]
    riscos += [f"{pid};R1;planejamento;Risco real do processo {pid};;;;;1;nao" for pid in ids]
    (corpus / "corpus_riscos_reais.csv").write_bytes(("\r\n".join(riscos) + "\r\n").encode("utf-8-sig"))
    return corpus


def _resposta(prompt, marca="v1"):
    quantidade = int(re.search(r"Gere exatamente (\d+) riscos", prompt).group(1))
    categoria = re.search(r"## Categoria dos riscos\n(.+)", prompt).group(1).strip()
    return json.dumps(
        [
            {
                "titulo": f"Risco {marca} {categoria} {i}",
                "causa": f"Causa {i}",
                "consequencia": f"Consequência {i}",
                "probabilidade_padrao": 2,
                "impacto_padrao": 4,
                "acao_preventiva": "Prevenir",
                "acao_contingencia": "Contingenciar",
                "responsavel_sugerido": "Equipe",
            }
            for i in range(1, quantidade + 1)
        ],
        ensure_ascii=False,
    )


def _executar(corpus, provider, **kwargs):
    mensagens = []
    falharam = gerar.executar(
        corpus,
        provider,
        provedor="fake",
        modelo="modelo-fake",
        agora=lambda: datetime(2026, 11, 20, 10, 0, 0),
        saida=mensagens.append,
        **kwargs,
    )
    return falharam, mensagens


def _ler(caminho):
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


def _arquivos(pasta):
    return sorted(p.relative_to(pasta).as_posix() for p in pasta.rglob("*") if p.is_file())


# ---------------------------------------------------------------- leitura e validacao


def test_leitura_tolerante_e_filtro_incluir(tmp_path):
    linhas = [
        _linha("AAA-01", incluir="SIM", tipo="Aquisição"),
        _linha("BBB-01", incluir="não"),
        _linha("CCC-01", incluir=" Sim ", tipo="licenciamento", principal="nao"),
        _linha("DDD-01", incluir="", objeto=""),  # excluido: objeto vazio nao impede
    ]
    corpus = _corpus(tmp_path, linhas, encoding="cp1252", sep="\t", com_riscos=["AAA-01", "CCC-01"])
    processos = gerar.ler_processos(corpus)
    assert processos == [
        gerar.Processo("AAA-01", "Aquisição de certificados SSL A1", "aquisicao"),
        gerar.Processo("CCC-01", "Aquisição de certificados SSL A1", "software"),  # analise_principal nao filtra
    ]


def test_tipos_do_corpus_viram_codigos_do_dominio(tmp_path):
    corpus = _corpus(tmp_path, [_linha("A-01", tipo="aquisicao"), _linha("S-01", tipo="servico"), _linha("L-01", tipo="licenciamento")])
    assert [p.tipo_contratacao for p in gerar.ler_processos(corpus)] == ["aquisicao", "servico", "software"]


@pytest.mark.parametrize(
    ("linhas", "com_riscos", "mensagem"),
    [
        ([_linha("AAA-01"), _linha("BBB-01", objeto="")], None, "BBB-01: objeto vazio"),
        ([_linha("AAA-01"), _linha("BBB-01")], ["AAA-01"], "BBB-01: sem riscos"),
        ([_linha("AAA-01"), _linha("BBB-01", tipo="consultoria")], None, "BBB-01: tipo_sugerido 'consultoria'"),
        ([_linha("AAA-01", incluir="nao")], None, "Nenhum processo"),
    ],
)
def test_corpus_invalido_para_tudo_sem_gerar_nada(tmp_path, linhas, com_riscos, mensagem):
    corpus = _corpus(tmp_path, linhas, com_riscos=com_riscos)
    provider = FakeProvider(_resposta)
    with pytest.raises(gerar.GeracaoError, match=re.escape(mensagem)):
        _executar(corpus, provider)
    assert provider.calls == []
    assert not (corpus / "retrospectiva").exists()


def test_corpus_dentro_do_repositorio_e_recusado():
    with pytest.raises(gerar.GeracaoError, match="dentro do repositorio"):
        _executar(ROOT / "data", FakeProvider(_resposta))


# ---------------------------------------------------------------- casos e geracao


def test_cria_casos_com_valores_neutros_e_gera_riscos(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("LIC-01", objeto="Licenciamento de software Autodesk", tipo="licenciamento")])
    casos_csv = (corpus / "corpus_casos.csv").read_bytes()
    provider = FakeProvider(_resposta)

    falharam, _ = _executar(corpus, provider)

    assert falharam == 0 and len(provider.calls) == 2 * CHAMADAS
    assert (corpus / "corpus_casos.csv").read_bytes() == casos_csv
    assert _arquivos(corpus / "retrospectiva") == [
        "candidatos.csv",
        "casos/AAA-01.json",
        "casos/AAA-01_ia_pt.json",
        "casos/LIC-01.json",
        "casos/LIC-01_ia_pt.json",
        "geracao_log.csv",
    ]
    caso = json.loads((corpus / "retrospectiva" / "casos" / "LIC-01.json").read_text(encoding="utf-8"))
    assert caso == {
        "nome": "LIC-01",
        "objeto": "Licenciamento de software Autodesk",
        "tipo_contratacao": "software",
        "valor_estimado": 0,
        "criticidade": "media",
        "prazo": "",
        "modalidade": "",
        "contexto": "",
    }
    assert list(caso) == list(json.loads((ROOT / "data" / "casos" / "exemplo.json").read_text(encoding="utf-8")))
    nome, contexto = load_case(corpus / "retrospectiva" / "casos" / "LIC-01.json")  # formato do app
    assert (nome, contexto.tipo_contratacao, contexto.valor_estimado) == ("LIC-01", "software", 0.0)
    # o prompt recebe o objeto e os neutros, em portugues
    prompt = provider.calls[-1]
    assert "- Objeto: Licenciamento de software Autodesk" in prompt and "- Criticidade: media" in prompt
    assert "- Contexto adicional: \n" in prompt and "português do Brasil" in prompt

    cache = json.loads((corpus / "retrospectiva" / "casos" / "AAA-01_ia_pt.json").read_text(encoding="utf-8"))
    assert {categoria: len(itens) for categoria, itens in cache.items()} == ESPERADO


def test_log_registra_modelo_prompt_hash_quantidades_e_tentativas(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01")])
    _executar(corpus, FakeProvider(_resposta))
    [registro] = _ler(corpus / "retrospectiva" / "geracao_log.csv")
    assert list(registro) == gerar.LOG_FIELDS
    assert (registro["processo_id"], registro["provedor"], registro["modelo"]) == ("AAA-01", "fake", "modelo-fake")
    assert registro["versao_prompt"] == "v2" and registro["hash_biblioteca"] == gerar.library_hash(gerar.BIBLIOTECA)
    assert registro["data_hora"] == "2026-11-20T10:00:00"
    assert (registro["situacao"], registro["tentativas"], registro["falhas"]) == ("gerado", "1", "")
    assert {c: int(registro[f"qtd_{c}"]) for c in CATEGORIES} == ESPERADO
    assert int(registro["qtd_total"]) == TOTAL_IA


def test_nao_sobrescreve_cache_existente(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01")])
    _executar(corpus, FakeProvider(_resposta))
    cache = corpus / "retrospectiva" / "casos" / "AAA-01_ia_pt.json"
    antes = cache.read_bytes()

    outro = FakeProvider(lambda prompt: _resposta(prompt, marca="OUTRA GERACAO"))
    falharam, mensagens = _executar(corpus, outro)

    assert falharam == 0 and outro.calls == []
    assert cache.read_bytes() == antes
    assert len(_ler(corpus / "retrospectiva" / "geracao_log.csv")) == 1  # sem nova linha
    assert any("1 ja com cache; 0 a gerar" in m for m in mensagens)


def test_objeto_alterado_com_cache_existente_para_sem_sobrescrever(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01")])
    _executar(corpus, FakeProvider(_resposta))
    pasta = corpus / "retrospectiva" / "casos"
    antes = {p.name: p.read_bytes() for p in pasta.iterdir()}
    _corpus_novo = "\r\n".join([CABECALHO, _linha("AAA-01", objeto="Outro objeto")]) + "\r\n"
    (corpus / "corpus_casos.csv").write_bytes(_corpus_novo.encode("utf-8-sig"))

    provider = FakeProvider(_resposta)
    with pytest.raises(gerar.GeracaoError, match="AAA-01"):
        _executar(corpus, provider)
    assert provider.calls == [] and {p.name: p.read_bytes() for p in pasta.iterdir()} == antes


def test_categoria_incompleta_nao_grava_cache_e_o_caso_fica_pendente(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01")])

    def falha_em_gestao(prompt):
        categoria = re.search(r"## Categoria dos riscos\n(.+)", prompt).group(1)
        if "Gestão" in categoria:
            return "isto nao e json"
        return _resposta(prompt)

    provider = FakeProvider(falha_em_gestao)
    falharam, mensagens = _executar(corpus, provider, tentativas=2)

    assert falharam == 2 and len(provider.calls) == 2 * 2 * CHAMADAS
    assert _arquivos(corpus / "retrospectiva" / "casos") == ["AAA-01.json", "BBB-01.json"]  # sem cache
    registros = _ler(corpus / "retrospectiva" / "geracao_log.csv")
    assert [(r["processo_id"], r["situacao"], r["tentativas"], r["qtd_total"]) for r in registros] == [
        ("AAA-01", "falhou", "2", "0"),
        ("BBB-01", "falhou", "2", "0"),
    ]
    assert "tentativa 1: gestao:" in registros[0]["falhas"] and "tentativa 2: gestao:" in registros[0]["falhas"]
    assert _ler(corpus / "retrospectiva" / "candidatos.csv") == []
    assert any("[FALHA] AAA-01" in m for m in mensagens)

    falharam, _ = _executar(corpus, FakeProvider(_resposta))  # nova execucao gera os pendentes
    assert falharam == 0
    assert [r["situacao"] for r in _ler(corpus / "retrospectiva" / "geracao_log.csv")] == ["falhou", "falhou", "gerado", "gerado"]


def test_registra_tentativas_quando_a_segunda_da_certo(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01")])
    chamadas = {"n": 0}

    def primeira_falha(prompt):
        chamadas["n"] += 1
        return "{}" if chamadas["n"] == 1 else _resposta(prompt)

    falharam, _ = _executar(corpus, FakeProvider(primeira_falha))
    [registro] = _ler(corpus / "retrospectiva" / "geracao_log.csv")
    assert falharam == 0 and (registro["situacao"], registro["tentativas"]) == ("gerado", "2")
    assert registro["falhas"].startswith("tentativa 1: planejamento:")
    assert int(registro["qtd_total"]) == TOTAL_IA


def test_resposta_com_menos_riscos_que_o_pedido_conta_como_incompleta(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01")])

    def um_a_menos(prompt):
        itens = json.loads(_resposta(prompt))
        return json.dumps(itens[:-1] or itens)

    falharam, _ = _executar(corpus, FakeProvider(um_a_menos), tentativas=1)
    [registro] = _ler(corpus / "retrospectiva" / "geracao_log.csv")
    assert falharam == 1 and registro["situacao"] == "falhou" and "esperado 3" in registro["falhas"]
    assert not (corpus / "retrospectiva" / "casos" / "AAA-01_ia_pt.json").exists()


# ---------------------------------------------------------------- limite e dry-run


def test_dry_run_valida_sem_chamar_a_ia_nem_gravar(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01", tipo="servico")])
    provider = FakeProvider(_resposta)
    falharam, mensagens = _executar(corpus, provider, dry_run=True)
    assert falharam == 0 and provider.calls == []
    assert not (corpus / "retrospectiva").exists()
    assert any("[dry-run] BBB-01: caso novo; gerar; tipo=servico" in m for m in mensagens)
    assert any(f"{2 * CHAMADAS} chamada(s) de IA seriam feitas" in m for m in mensagens)
    # tambem roda sem provedor configurado
    assert gerar.executar(corpus, None, provedor="x", modelo="y", dry_run=True, saida=lambda _m: None) == 0


def test_limite_processa_so_os_primeiros_casos(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01"), _linha("CCC-01")])
    provider = FakeProvider(_resposta)
    _executar(corpus, provider, limite=2)
    assert len(provider.calls) == 2 * CHAMADAS
    assert _arquivos(corpus / "retrospectiva" / "casos") == [
        "AAA-01.json",
        "AAA-01_ia_pt.json",
        "BBB-01.json",
        "BBB-01_ia_pt.json",
    ]
    assert {l["processo_id"] for l in _ler(corpus / "retrospectiva" / "candidatos.csv")} == {"AAA-01", "BBB-01"}

    _executar(corpus, provider)  # depois, sem limite: so o que falta
    assert len(provider.calls) == 3 * CHAMADAS


# ---------------------------------------------------------------- candidatos


def test_candidatos_trazem_catalogo_aplicavel_e_riscos_de_ia(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01", tipo="aquisicao"), _linha("LIC-01", tipo="licenciamento")])
    _executar(corpus, FakeProvider(_resposta))

    bruto = (corpus / "retrospectiva" / "candidatos.csv").read_bytes()
    assert bruto.startswith(b"\xef\xbb\xbf" + ";".join(gerar.CANDIDATOS_FIELDS).encode())
    linhas = _ler(corpus / "retrospectiva" / "candidatos.csv")
    assert len({l["candidato_id"] for l in linhas}) == len(linhas)

    for processo, tipo in (("AAA-01", "aquisicao"), ("LIC-01", "software")):
        do_processo = [l for l in linhas if l["processo_id"] == processo]
        catalogo = [l for l in do_processo if l["origem"] == "catalogo"]
        ia = [l for l in do_processo if l["origem"] == "ia"]
        aplicaveis = [r for r in BIBLIOTECA if tipo in r.tipo_contratacao]
        assert [l["id_catalogo"] for l in catalogo] == [r.id for r in aplicaveis]
        assert len(ia) == TOTAL_IA and all(l["id_catalogo"] == "" for l in ia)
        assert {c: sum(1 for l in ia if l["categoria"] == c) for c in CATEGORIES} == ESPERADO
        assert all(l["titulo"] and l["categoria"] in CATEGORIES for l in do_processo)

    primeiro = next(l for l in linhas if l["processo_id"] == "AAA-01" and l["origem"] == "catalogo")
    risco = next(r for r in BIBLIOTECA if r.id == primeiro["id_catalogo"])
    assert primeiro["candidato_id"] == f"AAA-01_cat_{risco.id}"
    assert (primeiro["titulo"], primeiro["causa"]) == (" ".join(risco.titulo.split()), " ".join(risco.causa.split()))
    ia = next(l for l in linhas if l["candidato_id"] == "AAA-01_ia_planejamento_1")
    assert ia["titulo"].startswith("Risco v1 ") and ia["causa"] == "Causa 1" and ia["consequencia"] == "Consequência 1"


def test_script_nao_altera_o_repositorio(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01")])
    antes = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    biblioteca = gerar.BIBLIOTECA.read_bytes()
    _executar(corpus, FakeProvider(_resposta))
    assert gerar.BIBLIOTECA.read_bytes() == biblioteca
    assert subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout == antes


def test_recusa_geral_do_provedor_interrompe_a_execucao(tmp_path):
    corpus = _corpus(tmp_path, [_linha("AAA-01"), _linha("BBB-01"), _linha("CCC-01")])
    provider = FakeProvider(fail=True)  # todas as chamadas falham com o mesmo erro (saldo, chave, rede)

    falharam, mensagens = _executar(corpus, provider)

    assert falharam == 1 and len(provider.calls) == CHAMADAS  # um caso, uma tentativa
    [registro] = _ler(corpus / "retrospectiva" / "geracao_log.csv")
    assert (registro["processo_id"], registro["situacao"], registro["tentativas"]) == ("AAA-01", "falhou", "1")
    assert registro["falhas"] == "tentativa 1: todas as categorias: Falha simulada do FakeProvider."
    assert any("execucao interrompida (2 caso(s) nem foram tentados)" in m for m in mensagens)
    assert not list((corpus / "retrospectiva" / "casos").glob("*_ia_*.json"))


def test_resumo_do_erro_tira_o_identificador_da_requisicao():
    erro = "Error code: 400 - {'type': 'error', 'error': {'message': 'saldo'}, 'request_id': 'req_011abc'}"
    assert gerar._resumir_erro(erro) == "Error code: 400 - {'type': 'error', 'error': {'message': 'saldo'}}"