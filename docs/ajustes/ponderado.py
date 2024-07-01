# %% [markdown]
# # Ajuste ponderado
#
# Si los diferentes valores de `y` tienen asociada una incerteza distinta,
# podemos hacer un ajuste ponderado,
# para que `curve_fit` le de mayor peso a los valores con menor incerteza.

# %%
import numpy as np
from labo1 import curve_fit


def lineal(x, a, b):
    return a * x + b


x = np.arange(10)
y = 2 * x + 3
y = np.random.default_rng(0).normal(y)
y[-1] -= 20  # resta 20 al último valor

result = curve_fit(lineal, x, y, estimate_errors=True)
result.plot()
result

# %% [markdown]
# Podemos darle menor importancia a la última medición
# pasando un vector de incertezas o errores `y_err`:

# %%
y_err = np.ones_like(y)  # [1, 1, ..., 1, 1]
y_err[-1] = 10           # [1, 1, ..., 1, 10]

result = curve_fit(lineal, x, y, y_err)
result.plot()
result


# %% [markdown]
# ### Reescalado de errores
#
# Cuando no conocemos la incerteza real de las mediciones,
# pero sabemos que son distintas entre sí,
# podemos combinar el vector de incertezas `y_err` con `estimate_errors`.
# Esto asume que conocemos la relación o cociente entre incertezas,
# Por ejemplo,
# si el vector de errores fuese `y_err = [2, 4, ...]`,
# interpretaría que la segunda medición tiene el doble de error que la primera,
# pero usa una versión reescalada de estos para estimar los errores en los parámetros.

# %%
curve_fit(lineal, x, y, y_err, estimate_errors=True)

# %% [markdown]
# El factor de reescalado es tal que $\chi^2$ reducido es 1.
