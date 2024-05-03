from ..round import to_significant_figures


def test_round():
    assert to_significant_figures(0.1234, n=2) == "0.12"


def test_round_error():
    assert to_significant_figures(12.34, 5.678, n=2) == ("12.3", "5.7")


def test_zero():
    assert to_significant_figures(0, n=2) == "0.0"
