from typing import overload

import numpy as np


@overload
def to_significant_figures(
    x: float,
    dx: None = None,
    /,
    n: int = 2,
) -> str: ...


@overload
def to_significant_figures(
    x: float,
    dx: float,
    /,
    n: int = 2,
) -> tuple[str, str]: ...


def to_significant_figures(
    x: float,
    dx: float | None = None,
    /,
    n: int = 2,
):
    """Rounds to `n` significant figures based on the uncertainty `dx`.

    Parameters:
        n: Number of significant figures.

    Examples:
        >>> to_significant_figures(0.1234, n=2)
        '0.12'
        >>> to_significant_figures(12.34, n=2)
        '12'
        >>> to_significant_figures(1234, n=2)
        '1200'
        >>> to_significant_figures(12.34, 5.678, n=2)
        ('12.3', '5.7')
    """
    if dx is None:
        return to_significant_figures(x, x, n=n)[0]

    if dx == 0:
        decimals = n - 1
    else:
        decimals = n - np.ceil(np.log10(dx)).astype(int)

    if decimals < 0:
        x = round(x, decimals)
        dx = round(dx, decimals)
        decimals = 0

    return f"{x:.{decimals}f}", f"{dx:.{decimals}f}"
