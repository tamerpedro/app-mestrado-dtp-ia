from zipfile import ZipFile

from src.docx_exporter import to_docx
from src.models import ActionItem, ContractContext, MatrixRow


def test_docx_export_contains_risk_map_structure():
    context = ContractContext(
        objeto="Aquisição de solução de TIC",
        tipo_contratacao="aquisicao",
        valor_estimado=1000,
        criticidade="media",
        prazo="12 meses",
        modalidade="pregao",
        contexto="Contratação de TIC",
    )
    rows = [
        MatrixRow(
            id="R001",
            risco="Especificação técnica insuficiente",
            categoria="planejamento",
            causa="Levantamento incompleto",
            consequencias=["Contratação inadequada", "Atraso na entrega"],
            probabilidade=3,
            impacto=4,
            nivel="high",
            estrategia="mitigate",
            acoes_preventivas=[
                ActionItem("Validar requisitos", situacao="in_progress", responsavel="Equipe de planejamento"),
                ActionItem("Revisar artefatos", situacao="completed", responsavel="Fiscal técnico"),
            ],
            acoes_contingencia=[
                ActionItem("Revisar especificações", situacao="not_started", responsavel="Equipe de planejamento")
            ],
        )
    ]

    content = to_docx(rows, context)

    with ZipFile(__import__("io").BytesIO(content)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    assert "MAPA DE GERENCIAMENTO DE RISCOS" in document_xml
    assert "Riscos do Planejamento" in document_xml
    assert "AÇÕES PREVENTIVAS" in document_xml
    assert "Atraso na entrega" in document_xml
    assert "Iniciado" in document_xml
    assert "Concluído" in document_xml
    assert "Não iniciado" in document_xml
    assert "3-Média" in document_xml
    assert "8 a 12 - Alto" in document_xml
    assert "Mitigar" in document_xml
    assert "in_progress" not in document_xml


def test_docx_section_numbers_go_from_1_to_6_without_repeating():
    """Bug real: Solucao e Instalacao ficavam as duas com "4 -", e Cronograma
    com "5 -" (deveria ser 6). A numeracao das secoes deve seguir 1..6 na
    ordem das categorias, sem repetir nenhum numero."""
    context = ContractContext(
        objeto="Aquisição de solução de TIC",
        tipo_contratacao="aquisicao",
        valor_estimado=1000,
        criticidade="media",
        prazo="12 meses",
        modalidade="pregao",
        contexto="Contratação de TIC",
    )

    def _row(id_: str, categoria: str) -> MatrixRow:
        return MatrixRow(
            id=id_,
            risco=f"Risco {id_}",
            categoria=categoria,
            causa="Causa",
            consequencias=["Consequencia"],
            probabilidade=3,
            impacto=3,
            nivel="moderado",
            estrategia="mitigar",
            acoes_preventivas=[ActionItem("Preventiva")],
            acoes_contingencia=[ActionItem("Contingencia")],
        )

    rows = [
        _row("R001", "planejamento"),
        _row("R002", "selecao"),
        _row("R003", "gestao"),
        _row("R004", "solucao"),
        _row("R005", "instalacao"),
        _row("R006", "cronograma"),
    ]

    content = to_docx(rows, context)

    with ZipFile(__import__("io").BytesIO(content)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    assert "1 - Riscos do Planejamento da Contratação" in document_xml
    assert "2 - Riscos da Seleção de Fornecedor" in document_xml
    assert "3 - Riscos da Gestão do Contrato" in document_xml
    assert "4 - Riscos da Solução" in document_xml
    assert "5 - Riscos da Instalação dos Equipamentos" in document_xml
    assert "6 - Riscos para o cronograma da contratação" in document_xml
