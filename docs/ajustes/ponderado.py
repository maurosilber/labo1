# %% [markdown]
# # Ajuste ponderado
#
# Por defecto,
# `curve_fit` asume que la incerteza de las mediciones son todos iguales,
# y las estima a partir de los residuos.
# Sin embargo,
# es importante incluir las incertezas cuando estas no son todas iguales:

# %%
import numpy as np
from labo1 import curve_fit


def lineal(x, a, b):
    return a * x + b


x = np.arange(10)
y = 2 * x + 3
y[-1] -= 20  # resta 20 al último valor
y = np.random.default_rng(0).normal(y)

result = curve_fit(lineal, x, y)
result.plot()
result

# %% [markdown]
# Podemos darle menor importancia a la ültima medición
# pasando un vector de incertezas o errores `y_err`:

# %%
y_err = np.ones(y.size)
y_err[-1] = 10

result = curve_fit(lineal, x, y, y_err)
result.plot()
result

# %% [markdown]
# De esta manera, también realiza el gráfico con barras de errores.

# %% [markdown]
# ### Reescalado de errores
#
# Por defecto,
# `curve_fit` estima los errores a partir de los residuos del ajuste.
# Esto sucede incluso si le pasamos las incertezas explícitamente.
#
# Por ejemplo,
# si el vector de errores fuese `y_err = [2, 4, ...]`,
# interpretaría que la segunda medición tiene el doble de error que la primera,
# pero usa una versión reescalada de estos para estimar los errores en los parámetros.
#
# Podemos ver esto al pasar el vector de errores anterior:

# %%
curve_fit(lineal, x, y, y_err)

# %% [markdown]
# y el mismo dividido 100:

# %%
curve_fit(lineal, x, y, y_err / 100)

# %% [markdown]
# En ambos casos,
# el error en los parámetros es el mismo.
#
# Pero,
# si conocemos los errores reales y no queremos que los reescale,
# podemos pedirselo con `rescale_errors=False`:

# %%
curve_fit(lineal, x, y, y_err / 100, rescale_errors=False)
