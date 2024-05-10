from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
from warnings import warn

import numpy as np
import scipy.optimize
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from .plot import with_errorbars, with_residuals
from .round import to_significant_figures


@dataclass(frozen=True)
class Result:
    func: Callable[..., np.ndarray]
    params: np.ndarray
    covariance: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray | None

    @property
    def names(self) -> Sequence[str]:
        """Names of the parameters.

        Extracted from the function signature."""
        return _get_parameter_names(self.func)

    @property
    def errors(self) -> np.ndarray:
        """Standard deviation for the parameters.

        The square-root of the diagonal of the covariance matrix."""
        return np.sqrt(np.diag(self.covariance))

    def __getitem__(self, item: int | str) -> tuple[float, float]:
        if isinstance(item, str):
            item = self.names.index(item)
        return self.params[item], self.errors[item]

    def plot(
        self,
        *,
        x_eval: int | np.ndarray | None = None,
        x_err: np.ndarray | None = None,
        label: str | None = None,
        fig: Figure | SubFigure | None = None,
        axes: Axes | None = None,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Errorbar plot of the data and line plot of the function.

        Parameters:
            x_eval: Evaluation points for the line plot of the function.
            For an `int`, it generates equispaced points between the minimum and maximum of `x`.
            By default, `x_eval = x`.
            x_err: Error bars for `x`.
            label: Name of the line plot for the legend.
            axes: Axes on which to plot.
            By default, creates a new axes on `fig`.
            fig: Figure on which to create the `axes`.
            By default, creates a new figure.

        Returns:
            The axes on which it plotted and its corresponding figure.
        """
        return with_errorbars(
            self.func,
            self.params,
            self.x,
            self.y,
            y_err=self.y_err,
            x_err=x_err,
            x_eval=x_eval,
            label=label,
            fig=fig,
            axes=axes,
        )

    def plot_with_residuals(
        self,
        *,
        x_eval: np.ndarray | None = None,
        x_err: np.ndarray | None = None,
        label: str | None = None,
        fig: Figure | SubFigure | None = None,
        axes: Sequence[Axes] | None = None,
    ) -> tuple[Figure | SubFigure, Sequence[Axes]]:
        """Errorbar plot of the data and residuals, and line plot of the function.

        Parameters:
            x_eval: Evaluation points for the line plot of the function.
            For an `int`, it generates equispaced points between the minimum and maximum of `x`.
            By default, `x_eval = x`.
            x_err: Error bars for `x`.
            label: Name of the line plot for the legend.
            axes: Axes on which to plot.
            By default, creates a new axes on `fig`.
            fig: Figure on which to create the `axes`.
            By default, creates a new figure.

        Returns:
            The axes on which it plotted and its corresponding figure.
        """
        return with_residuals(
            self.func,
            self.params,
            self.x,
            self.y,
            y_err=self.y_err,
            x_err=x_err,
            x_eval=x_eval,
            label=label,
            fig=fig,
            axes=axes,
        )

    def __str__(self):
        values = {
            name: to_significant_figures(x, dx)
            for name, x, dx in zip(self.names, self.params, self.errors)
        }
        values = [f"{name}={x} ± {dx}" for name, (x, dx) in values.items()]
        return ", ".join(values)

    def __repr__(self):
        return f"{self.__class__.__name__}({self})"


def curve_fit(
    func: Callable[..., np.ndarray],
    /,
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None = None,
    *,
    initial_params: Sequence[float] | Mapping[str, float] | None = None,
    rescale_errors: bool = True,
    **kwargs,
):
    """Use non-linear least squares to fit a function to data.

    Returns a `Result` object with the parameters, errors,
    and methods to quickly plot the fit.

    Parameters:
        func: The function to fit. Its signature must start with the independent variable `x`
        followed by its N parameters to fit: `f(x, p_0, p_1, ...)`.
        y_err: Errors or uncertainties for `y`.
        initial_params: Initial guess for the parameters.
        A sequence of length N or a mapping of names to values,
        where omitted values default to 1.
        rescale_errors: Whether to estimate a scale factor for the errors based on the residuals.
        **kwargs: Passed to scipy.optimize.curve_fit.

    Examples:
        >>> def f(x, a, b):
        ...     return a * x + b
        ...
        >>> x = np.array([0.0, 1.0, 2.0])
        >>> y = np.array([0.0, 0.9, 2.1])
        >>> curve_fit(f, x, y)
        Result(a=1.050 ± 0.087, b=-0.05 ± 0.11)
    """
    if isinstance(initial_params, Mapping):
        names = _get_parameter_names(func)
        unused_params = initial_params.keys() - names
        if len(unused_params) > 0:
            warn(f"unused parameters: {unused_params}")

        initial_params = [initial_params.get(name, 1) for name in names]  # type: ignore

    p, cov = scipy.optimize.curve_fit(
        func,
        x,
        y,
        p0=initial_params,
        sigma=y_err,
        absolute_sigma=not rescale_errors,
        **kwargs,
    )
    return Result(func, p, cov, x=x, y=y, y_err=y_err)


def _get_parameter_names(func: Callable) -> Sequence[str]:
    return list(inspect.signature(func).parameters)[1:]
