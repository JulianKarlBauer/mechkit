# # Notation: Converter Symbolic

import mechkit
import numpy as np
import sympy as sp
import itertools

np.set_printoptions(
    linewidth=140,
    precision=3,
    # suppress=False,
)

# ### Symbolic with numbers

converter = mechkit.notation.ConverterSymbolic()
ones_tensor = np.ones((3, 3, 3, 3), dtype=sp.Symbol)
print(ones_tensor)

ones_mandel6 = converter.to_mandel6(ones_tensor)
print(ones_mandel6)

ones_mandel9 = converter.to_mandel9(ones_tensor)
print(ones_mandel9)

# ### Symbolic with letters


def tensor(
    order=2, symbol="A", dim=3, latex_index=False, kwargs_symbol={}, indice_offset=0
):
    A = np.zeros((dim,) * order, dtype=sp.Symbol)
    for x in itertools.product(range(dim), repeat=order):
        index = "".join(map(str, map(lambda x: x + indice_offset, x)))
        if latex_index:
            index = "_{" + index + "}"
        A[x] = sp.Symbol(symbol + index, **kwargs_symbol)
    return A


def make_it_hooke_symmetric(A, dim=3):
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    A[i, j, m, k] = A[i, j, k, m]
                    A[j, i, m, k] = A[i, j, k, m]
                    A[k, m, i, j] = A[i, j, k, m]
    return A


def make_it_left_symmetric(A, dim=3):
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    A[j, i, k, m] = A[i, j, k, m]
    return A


def make_it_right_symmetric(A, dim=3):
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    A[i, j, m, k] = A[i, j, k, m]
    return A


def make_it_minor_symmetric(A, dim=3):
    tmp = make_it_left_symmetric(A)
    tmp = make_it_right_symmetric(A)
    return tmp


tensor = make_it_minor_symmetric(tensor(order=4, indice_offset=1))
print(tensor)

tensor_mandel6 = converter.to_mandel6(tensor)
print(tensor_mandel6)

tensor_mandel9 = converter.to_mandel9(tensor)
print(tensor_mandel9)
