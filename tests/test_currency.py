from src.currency import format_brl, parse_brl


def test_format_brl_adds_thousand_separators_and_decimal_comma():
    assert format_brl(15000000.0) == "15.000.000,00"


def test_format_brl_small_value():
    assert format_brl(159000.0) == "159.000,00"


def test_format_brl_zero():
    assert format_brl(0.0) == "0,00"


def test_parse_brl_reads_formatted_text():
    assert parse_brl("15.000.000,00") == 15000000.0


def test_parse_brl_reads_plain_digits_without_separators():
    assert parse_brl("15000000") == 15000000.0


def test_parse_brl_reads_cents():
    assert parse_brl("1.234,5") == 1234.5


def test_parse_brl_ignores_currency_prefix_and_spaces():
    assert parse_brl("R$ 159.000,00") == 159000.0


def test_parse_brl_empty_text_is_zero():
    assert parse_brl("") == 0.0
    assert parse_brl(None) == 0.0


def test_parse_brl_never_negative():
    assert parse_brl("-500") == 500.0


def test_format_and_parse_round_trip():
    for value in [0.0, 1.5, 159000.0, 15000000.0, 250000000.0]:
        assert parse_brl(format_brl(value)) == value
