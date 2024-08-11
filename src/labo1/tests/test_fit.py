import numpy as np
from pytest import mark, raises

from ..fit import curve_fit


def constant(x, A):
    return A


x = y = y_err = np.arange(1, 5)


def test_accept_ArrayLike():
    curve_fit(constant, list(x), list(y), list(y_err))


def test_initials_by_name():
    curve_fit(constant, x, y, y_err, initial_params={"A": 1})

    with raises(ValueError):
        curve_fit(constant, x, y, y_err, initial_params={"a": 1})


def test_do_not_estimate_errors_by_default():
    with raises(ValueError):
        curve_fit(constant, x, y)

    curve_fit(constant, x, y, estimate_errors=True)


@mark.parametrize("y_err", [1, y_err])
@mark.parametrize("estimate_errors", [False, True])
def test_estimate_errors(y_err, estimate_errors):
    curve_fit(constant, x, y, y_err, estimate_errors=estimate_errors)
