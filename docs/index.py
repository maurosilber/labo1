# %% [markdown]
# # Herramientas para laboratorio 1
#
# Este es un paquete de Python para facilitar el análisis de datos en Laboratorio 1.
#
# Se puede instalar con `pip`:
#
# ```
# pip install labo1
# ```
#
# En Google Colab,
# añadir un signo de exclamación al principio del comando:
#
# ```
# !pip install labo1
# ```

# %% [markdown]
# ## Cifras significativas
#
# Para redondear números y mediciones a una cantidad de cifras significativas,
# podemos usar:

# %%
from labo1 import to_significant_figures

# %% [markdown]
# Redondear un número a `n=2` cifras significativas:

# %%
to_significant_figures(0.00123456789, n=2)

# %% [markdown]
# Redondear una medición y su incerteza a `n=2` cifras significativas:

# %%
to_significant_figures(123.456789, 0.00123456789, n=2)

# %% [markdown]
# ## Ajustes por cuadrados mínimos
#
# Para hacer ajustes, podemos usar `curve_fit`.
#
# Para definir la función a ajustar,
# tenemos que poner primero la variable independiente `x`
# y luego los parámetros.
# En el ejemplo debajo,
# la función a ajustar es
#
# $$ f(x) = a x + b $$
#
# `curve_fit` nos devuelve un objeto con el resultado del ajuste
# con el cuál podemos realizar rápidamente un gráfico:

# %%
import numpy as np
from labo1 import curve_fit

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.4, 5.3, 6.6, 9.6, 11.0])


def lineal(x, a, b):
    return a * x + b


result = curve_fit(lineal, x, y)
result.plot()
result

# %% [markdown]
# En las siguientes secciones,
# hay más detalles sobre las funcionalidades de `curve_fit`:
#
# - [Básico](ajustes/basico): como acceder a los parámetros, errores y graficar los residuos.
# - [Ponderado](ajustes/ponderado): como ponderar los errores en un ajuste.
# - [No lineal](ajustes/no-lineal): como pasar parámetros iniciales a un ajuste.
# - [Gráficos](ajustes/graficos): como realizar gráficos más complejos.
