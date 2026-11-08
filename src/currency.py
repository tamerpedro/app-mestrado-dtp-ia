"""Formatacao/leitura de valores em reais com separador de milhar.

``st.number_input`` nao e pratico para contratacoes de centenas de milhoes
de reais (as setas de +/- sobem de 1 em 1 unidade de passo); o campo de
valor estimado usa texto livre com esse formato em vez disso.
"""

from __future__ import annotations

import re

_NON_DIGIT_OR_COMMA_RE = re.compile(r"[^0-9,]")


def format_brl(value: float) -> str:
    """``15000000.0`` -> ``"15.000.000,00"`` (separador de milhar ``.``, decimal ``,``)."""
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def parse_brl(text: str) -> float:
    """Inverso de ``format_brl``, tolerante a texto sem separadores (ex.: ``"15000000"``).

    Ignora qualquer caractere que nao seja digito ou virgula (prefixo "R$",
    espacos, pontos de milhar); a ultima virgula (se houver) marca os
    centavos. Nunca retorna negativo.
    """
    digits_and_comma = _NON_DIGIT_OR_COMMA_RE.sub("", text or "")
    if not digits_and_comma:
        return 0.0
    integer_part, sep, decimal_part = digits_and_comma.rpartition(",")
    if sep:
        decimal_part = (decimal_part + "00")[:2]
        cleaned = f"{integer_part or '0'}.{decimal_part}"
    else:
        cleaned = digits_and_comma
    try:
        return max(0.0, float(cleaned))
    except ValueError:
        return 0.0
