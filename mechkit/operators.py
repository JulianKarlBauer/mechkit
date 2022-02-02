#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import itertools
from mechkit import utils
from mechkit import notation
import functools

##########################################################################


def sym(tensor, sym_axes=None):
    """
    Symmetrize selected axes of tensor.
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


def is_sym(tensor, sym_axes=None):
    """Test whether `tensor` has the index symmetry specified by `sym_axes`"""
    return np.allclose(tensor, sym(tensor, sym_axes=sym_axes))


class Sym_Fourth_Order_Special(object):
    """
    Based on the `label` argument of the class initiation,
    the returned instance act as a symmetrization function,
    which symmetrices a given tensor with respect of the
    selected symmetry, following :cite:p:`Rychlewski2000`.
    """

    def __init__(self, label=None):

        self.set_permutation_lists()

        # Select the symmetrization operation
        if label is None:
            raise utils.Ex("Please specify a symmetry label")

        elif label == "inner_mandel":
            self.function = self._inner_mandel

        elif label in self.permutation_lists.keys():
            self.permutations = self._invert_permutations(self.permutation_lists[label])
            self.function = self._by_permutations

        else:
            raise utils.Ex("Please specify a valid symmetry label")

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def check(self, tensor, *args, **kwargs):
        return np.allclose(tensor, self.function(tensor=tensor, *args, **kwargs))

    def set_permutation_lists(self):
        base_permutations = {
            "identity": (0, 1, 2, 3),
            "left": (1, 0, 2, 3),
            "right": (0, 1, 3, 2),
            "major": (2, 3, 0, 1),
            "left_and_right": (1, 0, 3, 2),
            "<23>": [0, 2, 1, 3],
            "<24>": [0, 3, 2, 1],
        }

        self.permutation_lists = {
            "left": [base_permutations["identity"], base_permutations["left"]],
            "right": [base_permutations["identity"], base_permutations["right"]],
            "minor": [
                base_permutations["identity"],
                base_permutations["left"],
                base_permutations["right"],
                base_permutations["left_and_right"],
            ],
            "major": [base_permutations["identity"], base_permutations["major"]],
            # Following Rychlewski 2000 Equation (2.3) does not work
            # Follow suggestion https://physics.stackexchange.com/a/596930/175925
            "inner": [
                base_permutations["identity"],
                base_permutations["left"],
                base_permutations["right"],
                base_permutations["major"],
                base_permutations["left_and_right"],
                (3, 2, 0, 1),
                (2, 3, 1, 0),
                (3, 2, 1, 0),
            ],
            "complete": list(itertools.permutations([0, 1, 2, 3])),
        }

    def _invert_permutations(self, permutations):
        # Invert permutation according to Rychlewski2000 (A.8)
        inverted_perms = []
        for p in permutations:
            inverted_perms.append([p.index(0), p.index(1), p.index(2), p.index(3)])

        return inverted_perms

    def _by_permutations(self, tensor):
        return (
            1.0
            / len(self.permutations)
            * sum(tensor.transpose(perm) for perm in self.permutations)
        )

    def _inner_mandel(self, tensor):
        con = notation.Converter()
        t_mandel = con.to_mandel6(tensor)
        return con.to_like(
            inp=0.5 * (t_mandel + t_mandel.transpose()),
            like=tensor,
        )


##########################################################################


def dev_of_tensor_2_order(tensor):
    I2 = np.eye(3, dtype="float64")
    return tensor - 1.0 / 3.0 * I2 * np.einsum("...ii->...", tensor)


def dev_tensor_4_simple(tensor):
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
