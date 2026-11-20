"""Testes da consolidacao da avaliacao retrospectiva (Parte 3), com um corpus minimo e sem rede.

Corpus de exemplo (o que o Opus e o painel marcam):

- P1 (externo, servico, analise principal), riscos R1, R2, R3; candidatos cat_A
  (planejamento), cat_B (gestao), ia_1 (planejamento, grupo 1), ia_2 (gestao,
  grupo 1), ia_3 (gestao, grupo 2).
  Opus: R1 -> cat_A; R2 -> ia_1; R3 -> nada.
  Maioria: R1 -> cat_A; R2 -> nada; R3 -> ia_3.
- P2 (externo, aquisicao, fora da analise principal), riscos R1, R2; candidatos
  cat_A, ia_1 (grupo 1), ia_2 (grupo 2).
  Opus: R1 -> cat_A + ia_1; R2 -> nada. Maioria: R1 -> cat_A; R2 -> nada.
- P3 (dataprev, licenciamento, analise principal), risco R1; candidatos cat_A,
  ia_1. Opus e maioria: R1 -> cat_A.
- P4: incluir=nao.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import retro_conferencia as conf  # noqa: E402
import retro_consolidar as cons  # noqa: E402
import retro_gerar as gerar  # noqa: E402
import retro_julgar as julgar  # noqa: E402
import retro_painel as painel  # noqa: E402

CASOS = [
    ("P1", "externo", "servico", "sim", "sim"),
    ("P2", "externo", "aquisicao", "sim", "nao"),
    ("P3", "dataprev", "licenciamento", "sim", "sim"),
    ("P4", "externo", "servico", "nao", "nao"),
]
RISCOS = {"P1": ["R1", "R2", "R3"], "P2": ["R1", "R2"], "P3": ["R1"], "P4": ["R1"]}
# candidato: (origem, categoria); riscos de IA: (nomeia_objeto, termo, especifico_do_objeto, grupo)
CANDIDATOS = {
    "P1_cat_A": ("catalogo", "planejamento"),
    "P1_cat_B": ("catalogo", "gestao"),
    "P1_ia_1": ("ia", "planejamento"),
    "P1_ia_2": ("ia", "gestao"),
    "P1_ia_3": ("ia", "gestao"),
    "P2_cat_A": ("catalogo", "planejamento"),
    "P2_ia_1": ("ia", "solucao"),
    "P2_ia_2": ("ia", "solucao"),
    "P3_cat_A": ("catalogo", "planejamento"),
    "P3_ia_1": ("ia", "selecao"),
}
RISCOS_IA = {
    "P1_ia_1": ("sim", "firewall", "sim", "1"),
    "P1_ia_2": ("nao", "", "nao", "1"),
    "P1_ia_3": ("nao", "", "sim", "2"),
    "P2_ia_1": ("nao", "", "nao", "1"),
    "P2_ia_2": ("sim", "notebook", "nao", "2"),
    "P3_ia_1": ("nao", "", "nao", "1"),
}
OPUS = {
    ("P1", "R1"): ["P1_cat_A"],
    ("P1", "R2"): ["P1_ia_1"],
    ("P1", "R3"): [],
    ("P2", "R1"): ["P2_cat_A", "P2_ia_1"],
    ("P2", "R2"): [],
    ("P3", "R1"): ["P3_cat_A"],
}
PAINEL = {
    "anthropic": {**OPUS, ("P1", "R3"): ["P1_ia_3"], ("P2", "R1"): ["P2_cat_A"]},
    "openai": {**OPUS, ("P1", "R2"): [], ("P1", "R3"): ["P1_ia_3"], ("P2", "R1"): ["P2_cat_A"]},
    "google": {**OPUS, ("P1", "R2"): []},
}


def _julgamento(votos: dict) -> list[dict]:
    origem = {cid: dados[0] for cid, dados in CANDIDATOS.items()}
    return [
        {
            "processo_id": pid,
            "risco_id": rid,
            "candidatos_cobertos": julgar.SEPARADOR.join(marcados),
            "cobertura": julgar.derivar_cobertura([origem[c] for c in marcados]),
        }
        for (pid, rid), marcados in votos.items()
    ]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    saida = tmp_path / gerar.SAIDA_DIR
    (saida / julgar.PAINEL_DIR).mkdir(parents=True)
    conf.gravar_tabela(
        tmp_path / gerar.CASOS_CSV,
        ["processo_id", "grupo", "objeto", "tipo_sugerido", "incluir", "analise_principal"],
        [
            {"processo_id": p, "grupo": g, "objeto": f"objeto {p}", "tipo_sugerido": t, "incluir": i, "analise_principal": a}
            for p, g, t, i, a in CASOS
        ],
    )
    conf.gravar_tabela(
        tmp_path / gerar.RISCOS_CSV,
        ["processo_id", "risco_id", "fase", "risco", "causa", "consequencia"],
        [
            {"processo_id": p, "risco_id": r, "fase": "gestao", "risco": f"risco {p} {r}", "causa": f"causa {r}", "consequencia": "dano"}
            for p, lista in RISCOS.items()
            for r in lista
        ],
    )
    conf.gravar_tabela(
        saida / gerar.CANDIDATOS_CSV,
        gerar.CANDIDATOS_FIELDS,
        [
            {
                "processo_id": cid.split("_")[0],
                "candidato_id": cid,
                "origem": origem,
                "id_catalogo": cid.split("_")[-1] if origem == "catalogo" else "",
                "categoria": categoria,
                "titulo": f"titulo {cid}",
                "causa": f"causa {cid}",
                "consequencia": f"consequencia {cid}",
            }
            for cid, (origem, categoria) in CANDIDATOS.items()
        ],
    )
    conf.gravar_tabela(
        saida / julgar.RISCOS_IA_CSV,
        ["processo_id", "candidato_id", "categoria", "nomeia_objeto", "termo", "especifico_do_objeto", "grupo"],
        [
            {
                "processo_id": cid.split("_")[0],
                "candidato_id": cid,
                "categoria": CANDIDATOS[cid][1],
                "nomeia_objeto": nomeia,
                "termo": termo,
                "especifico_do_objeto": especifico,
                "grupo": grupo,
            }
            for cid, (nomeia, termo, especifico, grupo) in RISCOS_IA.items()
        ],
    )
    campos_log = ["processo_id", "modelo", "versao_prompt", "data_hora", "situacao", "qtd_total"]
    log = [
        # geracao antiga do P1 (substituida) e falha posterior (ignorada)
        {"processo_id": "P1", "modelo": "antigo", "versao_prompt": "v1", "data_hora": "2026-11-01T10:00:00", "situacao": "gerado", "qtd_total": "99"},
        {"processo_id": "P1", "modelo": "gerador", "versao_prompt": "v2", "data_hora": "2026-11-17T10:00:00", "situacao": "gerado", "qtd_total": "3"},
        {"processo_id": "P1", "modelo": "gerador", "versao_prompt": "v2", "data_hora": "2026-11-18T10:00:00", "situacao": "falhou", "qtd_total": "0"},
        {"processo_id": "P2", "modelo": "gerador", "versao_prompt": "v2", "data_hora": "2026-11-17T10:00:00", "situacao": "gerado", "qtd_total": "2"},
        {"processo_id": "P3", "modelo": "gerador", "versao_prompt": "v2", "data_hora": "2026-11-17T10:00:00", "situacao": "gerado", "qtd_total": "1"},
    ]  # fmt: skip
    conf.gravar_tabela(saida / gerar.LOG_CSV, campos_log, log)
    campos = ["processo_id", "risco_id", "candidatos_cobertos", "cobertura"]
    conf.gravar_tabela(saida / julgar.JULGAMENTO_CSV, campos, _julgamento(OPUS))
    for nome, votos in PAINEL.items():
        conf.gravar_tabela(julgar.arquivo_do_painel(saida, nome), campos, _julgamento(votos))
    painel.gravar_votos(saida, painel.analisar(tmp_path))
    # arquivos arquivados que a Parte 3 nao le: lixo de proposito
    (saida / "amostra_conferencia.csv").write_text("isto nao e uma tabela", encoding="utf-8")
    (saida / "adjudicacao_cobertura.csv").write_text("nem isto", encoding="utf-8")
    return tmp_path


def _grupo(resultado: dict, conjunto: str, grupo: str) -> dict:
    return next(g for g in resultado[conjunto] if g["grupo"] == grupo)


def _linha(tabela: list[dict], **filtro) -> dict:
    (linha,) = [l for l in tabela if all(l[campo] == valor for campo, valor in filtro.items())]
    return linha


def _hashes(pasta: Path) -> dict[str, str]:
    return {
        str(caminho): hashlib.sha256(caminho.read_bytes()).hexdigest()
        for caminho in pasta.rglob("*")
        if caminho.is_file() and cons.RESULTADOS_DIR not in caminho.parts
    }


def test_contagens_por_processo(corpus: Path) -> None:
    resultado = cons.consolidar(corpus)
    p1 = _linha(_grupo(resultado, "principal", "externo")["tabelas"]["por_processo"], processo_id="P1")
    esperado = {
        "tipo": "servico",
        "riscos_reais": 3,
        "so_catalogo": 1,
        "so_ia": 1,
        "ambos": 0,
        "nenhum": 1,
        "coberto": 2,
        "ia_gerados": 3,
        "ia_grupos": 2,
        "ia_nomeia_objeto": 1,
        "ia_especifico_do_objeto": 2,
        "ia_sem_correspondente": 2,  # ia_2 e ia_3
        "ia_grupos_sem_correspondente": 1,  # o grupo 1 tem ia_1, que tem correspondente
        "pct_coberto": "66,7",
        "modelo_gerador": "gerador",  # a linha "gerado" mais recente do log
        "versao_prompt_geracao": "v2",
    }
    assert {campo: p1[campo] for campo in esperado} == esperado

    p2 = _linha(_grupo(resultado, "sensibilidade", "externo")["tabelas"]["por_processo"], processo_id="P2")
    assert (p2["so_catalogo"], p2["so_ia"], p2["ambos"], p2["nenhum"]) == (0, 0, 1, 1)
    assert (p2["ia_gerados"], p2["ia_sem_correspondente"], p2["ia_grupos_sem_correspondente"]) == (2, 1, 1)
    assert p2["catalogo_com_ou_sem_ia"] == 1


def test_conjuntos_e_grupos_ficam_separados(corpus: Path) -> None:
    resultado = cons.consolidar(corpus)
    processos = {
        (conjunto, g["grupo"]): [linha["processo_id"] for linha in g["tabelas"]["por_processo"]]
        for conjunto, grupos in resultado.items()
        for g in grupos
    }
    assert processos == {
        ("principal", "externo"): ["P1"],
        ("principal", "dataprev"): ["P3"],
        ("sensibilidade", "externo"): ["P1", "P2"],
        ("sensibilidade", "dataprev"): ["P3"],
    }
    total = _grupo(resultado, "sensibilidade", "externo")["total"][painel.REFERENCIA]
    assert (total["riscos_reais"], total["so_catalogo"], total["so_ia"], total["ambos"], total["nenhum"]) == (5, 1, 1, 1, 2)
    assert (total["ia_gerados"], total["ia_sem_correspondente"]) == (5, 3)


def test_por_tipo(corpus: Path) -> None:
    tabela = _grupo(cons.consolidar(corpus), "sensibilidade", "externo")["tabelas"]["por_tipo"]
    assert [(l["tipo"], l["processos"], l["riscos_reais"], l["coberto"]) for l in tabela] == [
        ("aquisicao", 1, 2, 1),
        ("servico", 1, 3, 2),
    ]


def test_por_categoria(corpus: Path) -> None:
    tabela = _grupo(cons.consolidar(corpus), "principal", "externo")["tabelas"]["por_categoria"]
    assert [l["categoria"] for l in tabela] == ["planejamento", "gestao"]
    planejamento, gestao = tabela
    assert planejamento["candidatos_catalogo"] == 1
    assert planejamento["riscos_reais_cobertos_pelo_catalogo"] == 1  # R1, por cat_A
    assert planejamento["riscos_reais_cobertos_pela_ia"] == 1  # R2, por ia_1
    assert planejamento["riscos_reais_cobertos"] == 2
    assert (planejamento["ia_gerados"], planejamento["ia_sem_correspondente"], planejamento["ia_nomeia_objeto"]) == (1, 0, 1)
    assert gestao["riscos_reais_cobertos"] == 0
    assert (gestao["ia_gerados"], gestao["ia_sem_correspondente"], gestao["ia_especifico_do_objeto"]) == (2, 2, 1)


def test_listas_com_texto(corpus: Path) -> None:
    tabelas = _grupo(cons.consolidar(corpus), "sensibilidade", "externo")["tabelas"]
    nao_cobertos = tabelas["nao_cobertos"]
    assert [(l["processo_id"], l["risco_id"]) for l in nao_cobertos] == [("P1", "R3"), ("P2", "R2")]
    assert nao_cobertos[0]["risco"] == "risco P1 R3" and nao_cobertos[0]["causa"] == "causa R3"
    assert nao_cobertos[0]["cobertura_pela_maioria"] == "ia"

    sem = tabelas["ia_sem_correspondente"]
    assert [l["candidato_id"] for l in sem] == ["P1_ia_2", "P1_ia_3", "P2_ia_2"]
    ia_2, ia_3, notebook = sem
    assert (ia_2["grupo_ia"], ia_2["grupo_ia_sem_correspondente"]) == ("1", "nao")
    assert (ia_3["grupo_ia"], ia_3["grupo_ia_sem_correspondente"], ia_3["sem_correspondente_pela_maioria"]) == ("2", "sim", "nao")
    assert (notebook["nomeia_objeto"], notebook["termo"], notebook["titulo"]) == ("sim", "notebook", "titulo P2_ia_2")

    (incremental,) = tabelas["ia_cobertura_incremental"]
    assert (incremental["processo_id"], incremental["risco_id"], incremental["ia_id"]) == ("P1", "R2", "P1_ia_1")
    assert (incremental["risco"], incremental["ia_titulo"]) == ("risco P1 R2", "titulo P1_ia_1")
    assert (incremental["nomeia_objeto"], incremental["especifico_do_objeto"], incremental["termo"]) == ("sim", "sim", "firewall")


def test_robustez_bate_com_o_painel(corpus: Path) -> None:
    g = _grupo(cons.consolidar(corpus), "sensibilidade", "externo")
    analise = painel.analisar(corpus, processos={"P1", "P2"})
    tabelas = g["tabelas"]

    cobertura = {l["juiz"]: l for l in tabelas["robustez_cobertura"]}
    assert list(cobertura) == ["opus", "anthropic", "openai", "google", "maioria"]
    for nome, linha in cobertura.items():
        classes = analise["por_juiz"][nome]["classes"]
        assert (linha["so_catalogo"], linha["so_ia"], linha["ambos"], linha["nenhum"]) == (
            classes["catalogo"], classes["ia"], classes["ambos"], classes["nenhum"],
        )  # fmt: skip
    assert (cobertura["maioria"]["so_catalogo"], cobertura["maioria"]["so_ia"], cobertura["maioria"]["nenhum"]) == (2, 1, 2)

    concordancia = tabelas["robustez_concordancia"]
    geral = _linha(concordancia, medida="kappa de Cohen por par", a="maioria", b="opus")
    esperado = analise["comparacoes"][("maioria", "opus")]
    assert geral["n"] == esperado["pares"] == 21  # 3 riscos x 5 candidatos + 2 x 3
    assert geral["kappa"] == f"{esperado['kappa']:.4f}".replace(".", ",")
    assert len([l for l in concordancia if l["medida"] == "kappa de Cohen por par"]) == 7
    assert _linha(concordancia, medida="kappa de Fleiss por par")["kappa"] == f"{analise['fleiss']['kappa']:.4f}".replace(".", ",")
    mesmo = _linha(concordancia, medida="mesmo resultado coberto/nao coberto")
    assert (mesmo["n"], mesmo["iguais"], mesmo["concordancia"]) == (5, 3, "0,6000")
    assert _linha(concordancia, medida="mesma classe de cobertura")["iguais"] == 2

    p1 = _linha(tabelas["robustez_por_processo"], processo_id="P1")
    assert (p1["coberto_opus"], p1["coberto_maioria"], p1["riscos_com_mesmo_resultado"]) == (2, 2, 1)
    assert (p1["ia_sem_correspondente_opus"], p1["ia_sem_correspondente_maioria"]) == (2, 2)
    (ia,) = tabelas["robustez_ia_sem_correspondente"]
    assert (ia["ia_gerados"], ia["ia_sem_correspondente_opus"], ia["ia_sem_correspondente_maioria"]) == (5, 3, 4)
    assert (ia["pct_ia_sem_correspondente_opus"], ia["pct_ia_sem_correspondente_maioria"]) == ("60,0", "80,0")


def test_divergencias_entre_maioria_e_opus(corpus: Path) -> None:
    divergencias = _grupo(cons.consolidar(corpus), "sensibilidade", "externo")["tabelas"]["divergencias_maioria_opus"]
    # P2/R1 muda de classe (ambos -> catalogo) mas continua coberto: nao entra
    assert [(l["processo_id"], l["risco_id"]) for l in divergencias] == [("P1", "R2"), ("P1", "R3")]
    r2, r3 = divergencias
    assert (r2["cobertura_opus"], r2["cobertura_maioria"]) == ("ia", "nenhum")
    assert (r2["candidatos_opus"], r2["candidatos_maioria"], r2["candidatos_anthropic"], r2["candidatos_openai"]) == (
        "P1_ia_1", "", "P1_ia_1", "",
    )  # fmt: skip
    assert (r3["cobertura_opus"], r3["cobertura_maioria"], r3["cobertura_google"]) == ("nenhum", "ia", "nenhum")
    assert r3["candidatos_maioria"] == "P1_ia_3" and r3["titulos_dos_candidatos"] == "P1_ia_3: titulo P1_ia_3"
    assert r3["risco"] == "risco P1 R3"


def test_medianas_e_quartis(corpus: Path) -> None:
    assert cons.quartis([]) is None
    assert cons.quartis([0.5]) == (0.5, 0.5, 0.5)
    assert cons.quartis([0.0, 1.0]) == (0.25, 0.5, 0.75)
    assert cons.quartis([1.0, 2.0, 3.0, 4.0, 5.0]) == (2.0, 3.0, 4.0)
    medianas = _grupo(cons.consolidar(corpus), "sensibilidade", "externo")["medianas"][painel.REFERENCIA]
    # P1: 1/3 pelo catalogo, 2/3 no total; P2: 1/2 e 1/2
    assert medianas["cobertura do catalogo"][1] == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert medianas["cobertura total"][1] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert medianas["cobertura incremental da IA"][1] == pytest.approx(1 / 6)
    assert medianas["grupos distintos de riscos de IA"] == (2.0, 2.0, 2.0)


def test_gravar_cria_as_duas_pastas_e_nao_toca_nas_entradas(corpus: Path, capsys: pytest.CaptureFixture) -> None:
    antes = _hashes(corpus)
    assert cons.main(["--corpus", str(corpus)]) == 0
    assert _hashes(corpus) == antes
    resultados = corpus / gerar.SAIDA_DIR / cons.RESULTADOS_DIR
    esperados = {*cons.CAMPOS, "divergencias_maioria_opus"}
    for conjunto in cons.CONJUNTOS:
        assert {caminho.stem for caminho in (resultados / conjunto).iterdir()} == esperados
    por_processo = conf.ler_tabela(resultados / "sensibilidade" / "por_processo.csv", ["grupo", "processo_id"])
    assert [(l["grupo"], l["processo_id"]) for l in por_processo] == [("externo", "P1"), ("externo", "P2"), ("dataprev", "P3")]
    assert (resultados / "principal" / "por_tipo.csv").read_bytes().startswith(b"\xef\xbb\xbfgrupo;tipo;")

    saida = capsys.readouterr().out
    assert "ANALISE PRINCIPAL (incluir=sim e analise_principal=sim) | grupo externo: 1 processo(s), 3 riscos reais" in saida
    assert "SENSIBILIDADE (todos com incluir=sim) | grupo dataprev: 1 processo(s), 1 riscos reais" in saida
    assert "22 arquivo(s) gravado(s)" in saida
    assert "P4" not in saida


def test_relatorio_tem_cobertura_medianas_e_tipos(corpus: Path) -> None:
    texto = "\n".join(cons.relatorio(cons.consolidar(corpus)))
    assert "Opus (principal)" in texto and "maioria do painel" in texto
    assert "Sem correspondente no mapa real: Opus 3 (60,0%); maioria 4 (80,0%)" in texto
    assert "== Por processo: mediana [Q1 - Q3] ==" in texto
    assert "aquisicao (1) Opus" in texto and "servico (1) Opus" in texto
    assert "Mesmo resultado coberto/nao coberto: 3 (60,0%) de 5 riscos" in texto


def test_para_se_faltar_coluna(corpus: Path, capsys: pytest.CaptureFixture) -> None:
    caminho = corpus / gerar.SAIDA_DIR / julgar.RISCOS_IA_CSV
    linhas = conf.ler_tabela(caminho, [])
    conf.gravar_tabela(caminho, ["processo_id", "candidato_id", "categoria", "nomeia_objeto", "termo", "grupo"], linhas)
    with pytest.raises(conf.ConferenciaError, match="especifico_do_objeto"):
        cons.consolidar(corpus)
    assert cons.main(["--corpus", str(corpus)]) == 2
    assert "especifico_do_objeto" in capsys.readouterr().err
    assert not (corpus / gerar.SAIDA_DIR / cons.RESULTADOS_DIR).exists()


def test_para_se_o_log_nao_fecha_com_os_candidatos(corpus: Path) -> None:
    caminho = corpus / gerar.SAIDA_DIR / gerar.LOG_CSV
    campos = ["processo_id", "modelo", "versao_prompt", "data_hora", "situacao", "qtd_total"]
    nova = {"processo_id": "P2", "modelo": "gerador", "versao_prompt": "v2", "data_hora": "2026-11-18T12:00:00", "situacao": "gerado", "qtd_total": "17"}  # fmt: skip
    conf.gravar_tabela(caminho, campos, [nova], acrescentar=True)
    with pytest.raises(cons.ConsolidacaoError, match="P2: geracao_log.csv registra 17"):
        cons.consolidar(corpus)


def test_para_se_painel_votos_esta_desatualizado(corpus: Path) -> None:
    saida = corpus / gerar.SAIDA_DIR
    votos = {**PAINEL["google"], ("P2", "R2"): ["P2_ia_2"]}
    campos = ["processo_id", "risco_id", "candidatos_cobertos", "cobertura"]
    conf.gravar_tabela(julgar.arquivo_do_painel(saida, "google"), campos, _julgamento(votos))
    with pytest.raises(cons.ConsolidacaoError, match="painel_votos.csv nao confere"):
        cons.consolidar(corpus)


def test_para_se_falta_julgamento_de_algum_risco(corpus: Path) -> None:
    saida = corpus / gerar.SAIDA_DIR
    campos = ["processo_id", "risco_id", "candidatos_cobertos", "cobertura"]
    sem_p3 = {chave: marcados for chave, marcados in OPUS.items() if chave[0] != "P3"}
    for nome, votos in PAINEL.items():
        conf.gravar_tabela(julgar.arquivo_do_painel(saida, nome), campos, _julgamento({c: votos[c] for c in sem_p3}))
    with pytest.raises(cons.ConsolidacaoError, match="sem julgamento do Opus ou do painel"):
        cons.consolidar(corpus)
