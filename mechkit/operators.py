#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Currently only a small selection of operators and operator validation is public
"""

import numpy as np
import itertools


def sym(tensor, sym_axes=None):
    """Symmetrize selected axes of tensor
    If no sym_axes are specified, all axes are symmetrized
    """
    base_axis = np.array(range(len(tensor.shape)))

    sym_axes = base_axis if sym_axes is None else sym_axes

    perms = itertools.permutations(sym_axes)

    axes = list()
    for perm in perms:
        axis = base_axis.copy()
        axis[sym_axes] = perm
        axes.append(axis)

    return 1.0 / len(axes) * sum(tensor.transpose(axis) for axis in axes)


def dev_of_tensor_2_order(self, tensor):
    I2 = np.eye(3, dtype="float64")
    return tensor - 1.0 / 3.0 * I2 * np.einsum("...ii->...", tensor)


def dev_tensor_4_simple(self, tensor):
    assert tensor.shape == (3, 3, 3, 3,), (
        "Requires tensor 4.order in " "tensor notation"
    )

    tensor_4 = sym(tensor)
    tensor_2 = np.einsum("ppij->ij", tensor_4)
    trace = np.einsum("ii->", tensor_2)

    I2 = np.eye(3, dtype="float64")

    return (
        tensor_4
        - 6.0 / 7.0 * sym(np.multiply.outer(tensor_2, I2))
        + 3.0 / 35.0 * sym(np.multiply.outer(I2, I2)) * trace
    )


def dev(tensor, order=4):
    """Get deviatoric part of tensor"""
    # todo: make sure it is tensor

    functions = {
        2: dev_of_tensor_2_order,
        4: dev_tensor_4_simple,
    }

    return functions[order](tensor)
