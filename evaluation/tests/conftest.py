"""Riscos sinteticos pequenos e reutilizaveis, isolados de data/riscos_base.csv.

Mantem os testes do pacote evaluation independentes dos dados reais da
biblioteca (que podem mudar) e rapidos (poucos riscos, sem I/O de CSV).
"""

from __future__ import annotations

import pytest

from src.models import ContractContext, RiskItem


@pytest.fixture
def synthetic_risks() -> list[RiskItem]:
    return [
        RiskItem(
            id="S001",
            titulo="Risco de seguranca da informacao",
            categoria="planejamento",
            tipo_contratacao=["software", "servico"],
            palavras_chave=["seguranca", "dados", "acesso", "lgpd"],
            causa="Falta de requisitos de seguranca no termo de referencia",
            consequencia="Vazamento de dados sensiveis",
            probabilidade_padrao=3,
            impacto_padrao=5,
            acao_preventiva="Exigir certificacoes de seguranca",
            acao_contingencia="Acionar plano de resposta a incidentes",
            responsavel_sugerido="Equipe de seguranca",
        ),
        RiskItem(
            id="S002",
            titulo="Atraso no cronograma de implantacao",
            categoria="cronograma",
            tipo_contratacao=["aquisicao", "servico", "software"],
            palavras_chave=["prazo", "cronograma", "implantacao"],
            causa="Dependencia de terceiros nao mapeada",
            consequencia="Atraso na entrega da solucao",
            probabilidade_padrao=3,
            impacto_padrao=3,
            acao_preventiva="Definir marcos de acompanhamento",
            acao_contingencia="Aplicar plano de contingencia contratual",
            responsavel_sugerido="Gestor do contrato",
        ),
        RiskItem(
            id="S003",
            titulo="Especificacao tecnica insuficiente",
            categoria="planejamento",
            tipo_contratacao=["aquisicao", "servico", "software"],
            palavras_chave=["especificacao", "requisitos", "termo de referencia"],
            causa="Levantamento incompleto de requisitos",
            consequencia="Contratacao de solucao inadequada",
            probabilidade_padrao=3,
            impacto_padrao=4,
            acao_preventiva="Validar requisitos com areas usuarias",
            acao_contingencia="Revisar especificacoes e ajustar formalmente",
            responsavel_sugerido="Area demandante",
        ),
        RiskItem(
            id="S004",
            titulo="Preco estimado incompativel com o mercado",
            categoria="planejamento",
            tipo_contratacao=["aquisicao", "servico"],
            palavras_chave=["preco", "mercado", "orcamento"],
            causa="Pesquisa de precos insuficiente",
            consequencia="Licitacao deserta ou fracassada",
            probabilidade_padrao=2,
            impacto_padrao=3,
            acao_preventiva="Usar fontes diversificadas de precos",
            acao_contingencia="Refazer pesquisa de precos",
            responsavel_sugerido="Equipe de planejamento",
        ),
    ]


@pytest.fixture
def security_context() -> ContractContext:
    return ContractContext(
        objeto="Aquisicao de solucao de software de seguranca da informacao com homologacao",
        tipo_contratacao="software",
        valor_estimado=500000.0,
        criticidade="alta",
        prazo="12 meses",
        modalidade="pregao",
        contexto="Contratacao exige controle de acesso, lgpd e seguranca de dados sensiveis.",
    )
