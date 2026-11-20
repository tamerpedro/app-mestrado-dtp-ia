import pytest

from src.models import ActionItem, ContractContext, MatrixRow
from src.risk_library import load_risks, save_matrix_row_to_library


def test_load_risks_accepts_utf8_sig_header(tmp_path):
    path = tmp_path / "riscos_bom.csv"
    path.write_text(
        "\ufeffid,titulo,categoria,tipo_contratacao,palavras_chave,causa,consequencia,"
        "probabilidade_padrao,impacto_padrao,acao_preventiva,acao_contingencia,responsavel_sugerido\n"
        "R001,Risco teste,planejamento,software,requisitos,Causa,Consequencia,"
        "3-Média,4-Alto,Prevenir,Contingenciar,Equipe\n",
        encoding="utf-8",
    )

    risks = load_risks(path)

    assert risks[0].id == "R001"
    assert risks[0].titulo == "Risco teste"
    assert risks[0].probabilidade_padrao == 3
    assert risks[0].impacto_padrao == 4


def test_save_matrix_row_to_library_reloads_as_risk_item(tmp_path):
    path = tmp_path / "riscos_library.csv"
    context = ContractContext(
        objeto="Contratacao de software",
        tipo_contratacao="software",
        valor_estimado=1000,
        criticidade="media",
        prazo="12 meses",
        modalidade="pregao",
        contexto="Sistema corporativo",
    )
    row = MatrixRow(
        id="MAN001",
        risco="Falha de integracao",
        categoria="solucao",
        causa="Interfaces nao mapeadas",
        consequencias=["Indisponibilidade parcial"],
        probabilidade=3,
        impacto=4,
        nivel="high",
        estrategia="mitigate",
        acoes_preventivas=[ActionItem("Mapear interfaces", responsavel="Equipe tecnica")],
        acoes_contingencia=[ActionItem("Acionar plano de rollback", responsavel="Equipe tecnica")],
        tags=["manual"],
    )

    result = save_matrix_row_to_library(path, row, context)
    risks = load_risks(path)

    assert result.saved is True
    assert risks[0].id == "R001"
    assert risks[0].titulo == "Falha de integracao"
    assert risks[0].tipo_contratacao == ["software"]
    assert risks[0].responsavel_sugerido == ""
    assert (risks[0].probabilidade_padrao, risks[0].impacto_padrao) == (3, 4)
    assert "3-Média" in path.read_text(encoding="utf-8")


def test_load_risks_rejects_invalid_scale(tmp_path):
    path = tmp_path / "riscos_invalid.csv"
    path.write_text(
        "id,titulo,categoria,tipo_contratacao,palavras_chave,causa,consequencia,"
        "probabilidade_padrao,impacto_padrao,acao_preventiva,acao_contingencia,responsavel_sugerido\n"
        "R001,Risco teste,planejamento,software,requisitos,Causa,Consequencia,"
        "9,4-Alto,Prevenir,Contingenciar,Equipe\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="probabilidade_padrao"):
        load_risks(path)


def test_base_library_loads_with_numeric_scales():
    risks = load_risks("data/riscos_base.csv")

    assert len(risks) == 62
    assert all(risk.probabilidade_padrao in range(1, 6) for risk in risks)
    assert all(risk.impacto_padrao in range(1, 6) for risk in risks)


TRANSLATED = ["titulo", "causa", "consequencia", "acao_preventiva", "acao_contingencia", "responsavel_sugerido"]


def test_base_library_is_fully_translated_to_english():
    risks = load_risks("data/riscos_base.csv")

    for risk in risks:
        for field in TRANSLATED:
            assert risk.traducoes["en"].get(field), (risk.id, field)
        assert risk.palavras_chave_en, risk.id
    first = next(risk for risk in risks if risk.id == "R001")
    assert first.text("titulo", "en") == "Insufficient technical specification"
    assert first.text("titulo", "pt") == "Especificação técnica insuficiente"


def test_library_without_english_columns_falls_back_to_portuguese(tmp_path):
    path = tmp_path / "riscos_legacy.csv"
    path.write_text(
        "id,titulo,categoria,tipo_contratacao,palavras_chave,causa,consequencia,"
        "probabilidade_padrao,impacto_padrao,acao_preventiva,acao_contingencia,responsavel_sugerido\n"
        "R001,Risco teste,planejamento,software,requisitos,Causa,Consequencia,"
        "3-Média,4-Alto,Prevenir,Contingenciar,Equipe\n",
        encoding="utf-8",
    )

    risk = load_risks(path)[0]

    assert risk.text("titulo", "en") == "Risco teste"
    assert risk.palavras_chave_en == []


def _manual_row(title: str) -> MatrixRow:
    return MatrixRow(
        id="MAN001",
        risco=title,
        categoria="solucao",
        causa="Unmapped interfaces",
        consequencias=["Partial outage"],
        probabilidade=3,
        impacto=4,
        nivel="high",
        estrategia="mitigate",
        acoes_preventivas=[ActionItem("Map interfaces")],
        acoes_contingencia=[ActionItem("Roll back")],
        tags=["manual"],
    )


def _context() -> ContractContext:
    return ContractContext("Software procurement", "software", 1000, "media", "12 meses", "pregao", "Core system")


def test_risk_written_in_english_goes_to_english_columns(tmp_path):
    path = tmp_path / "riscos_en.csv"

    result = save_matrix_row_to_library(path, _manual_row("Integration failure"), _context(), lang="en")
    risk = load_risks(path)[0]

    assert result.saved is True
    assert risk.titulo == ""
    assert risk.traducoes["en"]["titulo"] == "Integration failure"
    assert risk.text("titulo", "pt") == "Integration failure"
    assert "integration" in risk.palavras_chave_en
    duplicate = save_matrix_row_to_library(path, _manual_row("Integration failure"), _context(), lang="en")
    assert duplicate.saved is False and duplicate.risk_id == "R001"


def test_saving_into_legacy_library_adds_english_columns(tmp_path):
    path = tmp_path / "riscos_upgrade.csv"
    path.write_text(
        "﻿id,titulo,categoria,tipo_contratacao,palavras_chave,causa,consequencia,"
        "probabilidade_padrao,impacto_padrao,acao_preventiva,acao_contingencia,responsavel_sugerido\r\n"
        "R001,Risco teste,planejamento,software,requisitos,Causa,Consequencia,"
        "3-Média,4-Alto,Prevenir,Contingenciar,Equipe\r\n",
        encoding="utf-8",
    )

    save_matrix_row_to_library(path, _manual_row("Integration failure"), _context(), lang="en")
    risks = load_risks(path)
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]

    assert [risk.id for risk in risks] == ["R001", "R002"]
    assert risks[0].titulo == "Risco teste"
    assert "titulo_en" in header
    assert b"\r\n" in path.read_bytes()
