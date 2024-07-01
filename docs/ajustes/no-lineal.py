# %% [markdown]
# # Funciones no lineales

# %% [markdown]
# Para ajustar funciones no lineales,
# suele ser necesario pasarle parámetros iniciales
# para que encuentre la solución correcta.
#
# Por ejemplo,
# si generamos datos que sigan una función:
#
# $$ f(x) = A \cos(\omega x) $$
#
# y tratamos de ajustarlos por esta función,
# vemos que no obtiene los parámetros correctos:

# %%
import numpy as np
from labo1 import curve_fit


def func(x, A, w):
    return A * np.cos(w * x)


x = np.linspace(0, 10, 100)
y = func(x, A=10, w=3)
y_err = 1

y = np.random.default_rng(0).normal(y, y_err)

r = curve_fit(func, x, y, y_err)
r.plot()
r

# %% [markdown]
# Si le pasamos parámetros iniciales cercanos a los reales,
# vemos que obtiene una mejor aproximación:

# %%
r = curve_fit(func, x, y, y_err, initial_params=(10, 3))
r.plot()
r

# %% [markdown]
# Al igual que los parámetros finales,
# los parámetros iniciales tienen que estar en el mismo orden
# que en el que se definió en la función.
#
# También,
# podemos pasar los parámetros por nombre con un diccionario,
# donde los parámetros omitidos toman de valor por defecto `1`:

# %%
curve_fit(func, x, y, y_err, initial_params={"w": 3})
