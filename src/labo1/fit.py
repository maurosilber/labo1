import inspect
from dataclasses import dataclass
from typing import Callable, Generic, Mapping, Sequence, TypeVarTuple
from warnings import warn

import numpy as np
import scipy.optimize
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from .plot import with_errorbars, with_residuals
from .round import to_significant_figures

P = TypeVarTuple("P")


@dataclass(frozen=True)
class Result(Generic[*P]):
    func: Callable[[np.ndarray, *P], np.ndarray]
    params: np.ndarray
    covariance: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_err: np.ndarray | None

    @property
    def names(self) -> Sequence[str]:
        return _get_parameter_names(self.func)

    @property
    def errors(self) -> np.ndarray:
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
    ):
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
    ):
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
    func: Callable[[np.ndarray, *P], np.ndarray],
    /,
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None = None,
    *,
    initial_params: tuple[*P] | Mapping[str, float] | None = None,
    rescale_errors: bool = True,
    **kwargs,
):
    """Use non-linear least squares to fit a function to data.

    Returns a Result object with the parameters, errors,
    and methods to quickly plot the fit.

    >>> def f(x, a, b):
    ...     return a * x + b
    >>> curve_fit(f, np.array([0, 1, 2]), np.array([0.1, 0.9, 2.1]))
    Result(a=1.00 ± 0.71, b=0.03 ± 0.91)
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
