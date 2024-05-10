# %% [markdown]
# # Graficando el ajuste

# %% [markdown]
# El objeto `Result` tiene dos métodos para realizar gráficos:
#
# - `.plot()`, que realiza un gráfico de las mediciones y el ajuste,
# - `.plot_with_residuals()`, que añade un gråfico de los residuos.
#
# En ambos casos,
# se incluye barras de error en `y` si fueron parte del ajuste.
#
# Además,
# aceptan los siguientes parámetros:
# - `x_err`, para gráficar barras de error en `x`,
# - `x_eval`, para evaluar la función ajustada en más puntos,
# - `label`, para darle un nombre a la curva del ajuste en la leyenda,
# - `fig` y `axes`, que permiten gráficar sobre un g®afico ya existente.
#
# Generemos y ajustemos unos datos para ver estas distintas opciones:

# %%
import numpy as np
from labo1 import curve_fit


def func(x, A, w):
    return A * np.cos(w * x)


x = np.linspace(0, 10, 10)
y = func(x, A=10, w=1)

y = np.random.default_rng(0).normal(y)

result = curve_fit(func, x, y)
result

# %% [markdown]
# ## Gráfico por defecto
#
# Este grafica la función para los mismos `x` que las mediciones.
# En este caso,
# vemos que no es suficiente para obtener una curva suave:

# %%
result.plot()

# %% [markdown]
# ## Evaluando en más puntos
#
# Se puede pedir que evalúe en más puntos pasando un número a `x_eval`:

# %%
result.plot(x_eval=100)

# %% [markdown]
# O un `array` con los valores a evaluar:

# %%
result.plot(x_eval=np.linspace(-5, 15, 100))

# %% [markdown]
# ## Múltiples gráficos
#
# El método `.plot` nos devuelve una figura y los ejes.
# Podemos realizar otro gráfico sobre los mismos ejes
# si se los pasamos al parámetro `axes` de la función.
# En este caso,
# es útil ponerle un nombre a cada curva con el parámetro `label`
# y generar una leyenda con `axes.legend`:

# %%
fig, axes = result.plot(x_eval=20, label="10")
result.plot(axes=axes, x_eval=100, label="100")
axes.legend(title="Ajuste")
