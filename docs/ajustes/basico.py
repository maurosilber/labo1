# %% [markdown]
# # Básico

# %% [markdown]
# Para realizar un ajuste,
# definimos la función a ajustar
# y se la pasamos a `curve_fit` junto con los datos:

# %%
import numpy as np
from labo1 import curve_fit


def lineal(x, a, b):
    return a * x + b


result = curve_fit(
    lineal,
    x=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    y=np.array([2.9, 5.0, 6.9, 9.0, 11.0]),
    estimate_errors=True,
)

# %% [markdown]
# donde usamos `estimate_errors=True` para que estime el error en `y`
# a partir de los residuos del ajuste.
# Ver la sección de [ajustes ponderados](../ponderado)
# si los errores son distintos para cada `y`
# o se conocen los errores.
#
# `result` es un objeto que encapsula el resultado del ajuste,
# mostrándonos los parámetros obtenidos con su error:

# %%
result

# %% [markdown]
# ### Acceder a los parámetros

# %% [markdown]
# Podemos acceder a ellos con:

# %%
result.params

# %% [markdown]
# y a sus incertezas con:

# %%
result.errors

# %% [markdown]
# El orden es el mismo que en la definición de la función:

# %%
lineal

# %% [markdown]
# También es posible acceder al valor y su incerteza por nombre:

# %%
result["a"]

# %% [markdown]
# ### Graficar el ajuste

# %% [markdown]
# Se puede realizar un gráfico rápidamente con:

# %%
result.plot()

# %% [markdown]
# El método `.plot` nos devuelve la figura y los ejes de `matplotlib`,
# a los cuales podemos agregarle nombres con:

# %%
fig, ax = result.plot()
ax.set(xlabel="eje x", ylabel="eje y")

# %% [markdown]
# En la sección [Graficando el ajuste](../graficos.py),
# hay más información sobre como graficar.

# %% [markdown]
# ## Graficar el ajuste con residuos

# %% [markdown]
# Para realizar un gráfico con residuos llamamos al método `.plot_with_residuals()`.
# A diferencia de `.plot`, este nos devuelve dos ejes:

# %%
fig, axes = result.plot_with_residuals()

axes[0].set(ylabel="y [unidad]")
axes[1].set(ylabel="Residuos", xlabel="x [unidad]")
fig.align_labels()
