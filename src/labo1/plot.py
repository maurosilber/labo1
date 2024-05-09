from typing import Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure


def with_errorbars(
    func,
    params: Sequence | np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_err: np.ndarray | None = None,
    y_err: np.ndarray | None = None,
    x_eval: int | np.ndarray | None = None,
    label: str | None = None,
    fig: Figure | SubFigure | None = None,
    axes: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    if axes is None:
        if fig is None:
            fig = plt.figure()
        axes = cast(Axes, fig.subplots())
    elif fig is not None:
        raise ValueError("specify either `fig` or `axes`")

    if x_eval is None:
        x_eval = x
    elif isinstance(x_eval, int):
        x_eval = np.linspace(np.min(x), np.max(x), x_eval)
        x_eval = cast(np.ndarray, x_eval)

    (line,) = axes.plot(x_eval, func(x_eval, *params), label=label)
    axes.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", color=line.get_color())
    fig = cast(Figure | SubFigure, axes.figure)
    return fig, axes


def with_residuals(
    func,
    params: Sequence | np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_err: np.ndarray | None = None,
    y_err: np.ndarray | None = None,
    x_eval: np.ndarray | None = None,
    label: str | None = None,
    fig: Figure | SubFigure | None = None,
    axes: Sequence[Axes] | None = None,
) -> tuple[Figure | SubFigure, Sequence[Axes]]:
    if axes is None:
        if fig is None:
            fig = plt.figure()
        axes = cast(
            Sequence[Axes],
            fig.subplots(nrows=2, sharex=True, height_ratios=[2, 1]),
        )
    elif fig is not None:
        raise ValueError("specify either `fig` or `axes`")
    elif len(axes) != 2:
        raise TypeError("`axes` must be a two-element sequence")

    if x_eval is None:
        x_eval = x

    residuals = y - func(x, *params)

    (line,) = axes[0].plot(x_eval, func(x_eval, *params), label=label)
    axes[1].axhline(0, color="gray")

    color = line.get_color()

    axes[0].errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", color=color)
    axes[1].errorbar(x, residuals, xerr=x_err, yerr=y_err, fmt="o", color=color)

    fig = cast(Figure | SubFigure, axes[0].figure)
    return fig, axes
