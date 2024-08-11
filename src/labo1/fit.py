from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence, Union, cast
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from numpy.typing import ArrayLike, NDArray

from .round import to_significant_figures


@dataclass(frozen=True)
class Result:
    func: Callable[..., NDArray]
    params: NDArray
    "Optimal parameters found by least squares."
    covariance: NDArray
    "Covariance matrix of the parameters."
    x: NDArray
    y: NDArray
    y_err: NDArray

    @property
    def names(self) -> Sequence[str]:
        """Names of the parameters.

        Extracted from the function signature."""
        return _get_parameter_names(self.func)

    @property
    def errors(self) -> NDArray:
        """Standard deviation for the parameters.

        The square-root of the diagonal of the covariance matrix."""
        return np.sqrt(np.diag(self.covariance))

    def __getitem__(self, item: int | str) -> tuple[float, float]:
        if isinstance(item, str):
            item = self.names.index(item)
        return self.params[item], self.errors[item]

    def eval(self, x: NDArray, /) -> NDArray:
        """Evaluates the function with the parameters."""
        return self.func(x, *self.params)

    @property
    def residuals(self):
        """The difference between the measured and predicted `y`.

        $$ r_i = y_i - f(x_i) $$
        """
        return self.y - self.eval(self.x)

    @property
    def standardized_residuals(self):
        """Residuals divided by their corresponding error."""
        return self.residuals / self.y_err

    @property
    def chi2(self):
        r"""Sum of the standardized squared residuals.

        $$ \chi^2 = \sum_i (\frac{r_i}{y_{err}_i})^2 $$
        """
        return np.sum(self.standardized_residuals**2)

    @property
    def reduced_chi2(self):
        """χ² divided by the degree of freedom.

        The degree of freedom is the number of measuments
        minus the number of fitted parameters.
        """
        return self.chi2 / (np.size(self.y) - np.size(self.params))

    def __str__(self):
        values = {
            name: to_significant_figures(x, dx)
            for name, x, dx in zip(self.names, self.params, self.errors)
        }
        values = [f"{name}={x} ± {dx}" for name, (x, dx) in values.items()]
        return ", ".join(values)

    def __repr__(self):
        return f"{self.__class__.__name__}({self})"

    def plot(
        self,
        *,
        x_err: ArrayLike | None = None,
        x_eval: int | ArrayLike | None = None,
        label: str | None = None,
        fig: Figure | SubFigure | None = None,
        axes: Axes | None = None,
    ) -> tuple[Figure | SubFigure, Axes]:
        """Errorbar plot of the data and line plot of the function.

        Parameters:
            x_eval: Evaluation points for the line plot of the function. For an `int`,
            it generates equispaced points between the minimum and maximum of `x`.
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
        if axes is None:
            if fig is None:
                fig = plt.figure()
            axes = cast(Axes, fig.subplots())
        elif fig is not None:
            raise ValueError("specify either `fig` or `axes`")

        if x_eval is None:
            x_eval = self.x
        elif isinstance(x_eval, int):
            x_eval = cast(NDArray, np.linspace(self.x.min(), self.x.max(), x_eval))
        else:
            x_eval = np.asarray(x_eval)

        (line,) = axes.plot(x_eval, self.eval(x_eval), label=label)
        axes.errorbar(
            self.x, self.y, xerr=x_err, yerr=self.y_err, fmt="o", color=line.get_color()
        )
        fig = cast(Union[Figure, SubFigure], axes.figure)
        return fig, axes

    def plot_with_residuals(
        self,
        *,
        x_err: ArrayLike | None = None,
        x_eval: int | ArrayLike | None = None,
        label: str | None = None,
        fig: Figure | SubFigure | None = None,
        axes: Sequence[Axes] | None = None,
    ) -> tuple[Figure | SubFigure, Sequence[Axes]]:
        """Errorbar plot of the data and residuals, and line plot of the function.

        Parameters:
            x_eval: Evaluation points for the line plot of the function. For an `int`,
            it generates equispaced points between the minimum and maximum of `x`.
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
        if axes is None:
            if fig is None:
                fig = plt.figure()
            axes = cast(
                Sequence[Axes],
                fig.subplots(
                    nrows=2,
                    sharex=True,
                    gridspec_kw={"height_ratios": [2, 1]},
                ),
            )
        elif fig is not None:
            raise ValueError("specify either `fig` or `axes`")
        elif len(axes) != 2:
            raise TypeError("`axes` must be a two-element sequence")

        if x_eval is None:
            x_eval = self.x
        elif isinstance(x_eval, int):
            x_eval = cast(NDArray, np.linspace(self.x.min(), self.x.max(), x_eval))
        else:
            x_eval = np.asarray(x_eval)

        residuals = self.y - self.eval(self.x)

        (line,) = axes[0].plot(x_eval, self.eval(x_eval), label=label)
        axes[1].axhline(0, color="gray")

        color = line.get_color()

        axes[0].errorbar(
            self.x, self.y, xerr=x_err, yerr=self.y_err, fmt="o", color=color
        )
        axes[1].errorbar(
            self.x, residuals, xerr=x_err, yerr=self.y_err, fmt="o", color=color
        )

        fig = cast(Union[Figure, SubFigure], axes[0].figure)
        return fig, axes


def curve_fit(
    func: Callable[..., NDArray],
    /,
    x: ArrayLike,
    y: ArrayLike,
    y_err: ArrayLike | None = None,
    *,
    initial_params: Sequence[float] | Mapping[str, float] | None = None,
    estimate_errors: bool = False,
    **kwargs,
):
    """Use non-linear least squares to fit a function to data.

    Returns a `Result` object with the parameters, errors,
    and methods to quickly plot the fit.

    Parameters:
        func: The function to fit. Its signature must start with the independent
        variable `x` followed by its N parameters to fit: `f(x, p_0, p_1, ...)`.
        y_err: Errors or uncertainties for `y`.
        initial_params: Initial guess for the parameters.
        A sequence of length N or a mapping of names to values,
        where omitted values default to 1.
        estimate_errors: Whether to estimate a global scale factor for the errors
        based on the residuals.
        **kwargs: Passed to scipy.optimize.curve_fit.

    Examples:
        >>> def f(x, a, b):
        ...     return a * x + b
        ...
        >>> x = np.array([0.0, 1.0, 2.0])
        >>> y = np.array([0.0, 0.9, 2.1])
        >>> curve_fit(f, x, y, estimate_errors=True)
        Result(a=1.050 ± 0.087, b=-0.05 ± 0.11)
    """
    # accept ArrayLike
    x = np.asarray(x)
    y = np.asarray(y)
    if y_err is not None:
        y_err = np.asarray(y_err)
    elif estimate_errors is False:
        raise ValueError("y_err cannot be None when estimate_errors is False.")
    else:
        y_err = np.ones_like(y)

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
        absolute_sigma=not estimate_errors,
        **kwargs,
    )
    r = Result(func, p, cov, x=x, y=y, y_err=y_err)
    if estimate_errors:
        # Errors have already been rescaled inside scipy's curve_fit
        # to estimate the parameters' errors.
        # Here, we rescale `y_err` to use later when plotting.
        r = replace(r, y_err=y_err * r.reduced_chi2**0.5)
    return r


def _get_parameter_names(func: Callable) -> Sequence[str]:
    return list(inspect.signature(func).parameters)[1:]
