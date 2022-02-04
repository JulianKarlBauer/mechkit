#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pprint
import mechkit
from mechkit.operators import Sym_Fourth_Order_Special
import pytest
import itertools

con = mechkit.notation.Converter()


@pytest.fixture(name="tensor4")
def create_random_fourth_order_tensor():
    return np.random.rand(3, 3, 3, 3)


def has_sym_left(A):
    for iii in range(3):
        for jjj in range(3):
            for kkk in range(3):
                for lll in range(3):
                    assert A[iii, jjj, kkk, lll] == A[jjj, iii, kkk, lll]


def has_sym_right(A):
    for iii in range(3):
        for jjj in range(3):
            for kkk in range(3):
                for lll in range(3):
                    assert A[iii, jjj, kkk, lll] == A[iii, jjj, lll, kkk]


def has_sym_major(A):
    for iii in range(3):
        for jjj in range(3):
            for kkk in range(3):
                for lll in range(3):
                    assert A[iii, jjj, kkk, lll] == A[kkk, lll, iii, jjj]


def has_sym_minor(A):
    for iii in range(3):
        for jjj in range(3):
            for kkk in range(3):
                for lll in range(3):
                    np.allclose(
                        A[iii, jjj, kkk, lll],
                        A[jjj, iii, kkk, lll],
                        rtol=1e-8,
                        atol=1e-8,
                    )
                    np.allclose(
                        A[iii, jjj, kkk, lll],
                        A[iii, jjj, lll, kkk],
                        rtol=1e-8,
                        atol=1e-8,
                    )


def has_sym_inner(A):
    for iii in range(3):
        for jjj in range(3):
            for kkk in range(3):
                for lll in range(3):
                    print(iii, jjj, kkk, lll)
                    assert np.isclose(A[iii, jjj, kkk, lll], A[jjj, iii, kkk, lll])
                    assert np.isclose(A[iii, jjj, kkk, lll], A[iii, jjj, lll, kkk])
                    assert np.isclose(A[iii, jjj, kkk, lll], A[kkk, lll, iii, jjj])


def has_sym_complete(A):
    for iii in range(3):
        for jjj in range(3):
            for kkk in range(3):
                for lll in range(3):
                    print(iii, jjj, kkk, lll)
                    assert np.isclose(A[iii, jjj, kkk, lll], A[jjj, iii, kkk, lll])
                    assert np.isclose(A[iii, jjj, kkk, lll], A[iii, jjj, lll, kkk])
                    assert np.isclose(A[iii, jjj, kkk, lll], A[kkk, lll, iii, jjj])
                    assert np.isclose(A[iii, jjj, kkk, lll], A[lll, kkk, jjj, iii])


class Test_Sym_Fourth_Order_Special:
    def test_check_sym_by_loop_left(self, tensor4):
        t_sym = Sym_Fourth_Order_Special(label="left")(tensor4)
        pprint.pprint(con.to_mandel9(t_sym))
        has_sym_left(t_sym)

    def test_check_sym_by_loop_right(self, tensor4):
        t_sym = Sym_Fourth_Order_Special(label="right")(tensor4)
        pprint.pprint(con.to_mandel9(t_sym))
        has_sym_right(t_sym)

    def test_check_sym_by_loop_major(self, tensor4):
        t_sym = Sym_Fourth_Order_Special(label="major")(tensor4)
        pprint.pprint(con.to_mandel9(t_sym))
        has_sym_major(t_sym)

    def test_check_sym_by_loop_minor(self, tensor4):
        t_sym = Sym_Fourth_Order_Special(label="minor")(tensor4)
        pprint.pprint(con.to_mandel9(t_sym))
        has_sym_minor(t_sym)

    def test_check_sym_by_loop_inner_mandel(self, tensor4):
        t_sym = Sym_Fourth_Order_Special(label="inner_mandel")(tensor4)
        pprint.pprint(con.to_mandel9(t_sym))
        has_sym_inner(t_sym)

    def test_check_sym_by_loop_inner(self, tensor4):
        t_sym = Sym_Fourth_Order_Special(label="inner")(tensor4)
        pprint.pprint(con.to_mandel9(t_sym))
        has_sym_inner(t_sym)


def test_check_sym_by_loop_complete(tensor4):
    t_sym = mechkit.operators.Sym()(tensor4)
    pprint.pprint(con.to_mandel9(t_sym))
    has_sym_complete(t_sym)


def test_check_sym_by_loop_alternative_implementation(tensor4):
    t_sym = mechkit.operators.Sym()(tensor4)

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

    np.allclose(sym(tensor4), t_sym)


def test_compare_sym_inner_inner_mandel(tensor4):
    t_sym_inner = Sym_Fourth_Order_Special(label="inner")(tensor4)
    t_sym_inner_mandel = Sym_Fourth_Order_Special(label="inner_mandel")(tensor4)
    print("t_sym_inner")
    pprint.pprint(con.to_mandel9(t_sym_inner))
    print("t_sym_inner_mandel")
    pprint.pprint(con.to_mandel9(t_sym_inner_mandel))
    np.allclose(t_sym_inner, t_sym_inner_mandel)


def test_sym_minor_mandel(tensor4):
    """Converting to Mandel6 and back should be identical to
    Sym(label=\'minor\')
    """
    t_sym_mandel = con.to_tensor(con.to_mandel6(tensor4))
    t_sym_label = Sym_Fourth_Order_Special(label="minor")(tensor4)

    print(con.to_mandel9(t_sym_mandel))
    print(con.to_mandel9(t_sym_label))

    assert np.allclose(t_sym_mandel, t_sym_label)


def test_sym_axes_label_left(tensor4):
    """Two implementation should do the same job"""
    sym_axes = mechkit.operators.Sym(axes=[0, 1])(tensor4)
    sym_label = Sym_Fourth_Order_Special(label="left")(tensor4)

    print(sym_axes)
    print(sym_label)

    assert np.allclose(sym_axes, sym_label)


def test_sym_axes_label_right(tensor4):
    """Two implementation should do the same job"""
    sym_axes = mechkit.operators.Sym(axes=[2, 3])(tensor4)
    sym_label = Sym_Fourth_Order_Special(label="right")(tensor4)

    print(sym_axes)
    print(sym_label)

    assert np.allclose(sym_axes, sym_label)


####################################################################


@pytest.fixture(name="tensors")
def create_random_tensors_with_minor_symmetries():
    tensors = {
        "hooke": mechkit.operators.Sym_Fourth_Order_Special(label="inner")(
            np.random.rand(3, 3, 3, 3)
        ),
        "complete": mechkit.operators.Sym_Fourth_Order_Special(label="complete")(
            np.random.rand(3, 3, 3, 3)
        ),
    }
    return tensors


alternatives = mechkit.operators.Alternative_Deviator_Formulations()


class Test_Deviators:
    def test_deviator_part_of_4_tensor_is_deviator(self, tensors):
        for key in ["hooke", "complete"]:
            deviator = alternatives.dev_t4_spencer1970(tensors[key])
            # Is symmetric
            sym = mechkit.operators.Sym()
            assert np.allclose(deviator, sym(deviator))
            # Is traceless (as it is already symmetric, one trace tested is sufficient)
            assert np.allclose(np.einsum("ijkk->ij", deviator), np.zeros((3, 3)))

    def test_deviator_part_of_4_tensor_implementations_spencer_boehlke(self, tensors):
        for key in ["hooke", "complete"]:
            D_spencer = alternatives.dev_t4_spencer1970(tensors[key])
            D_boehlke = alternatives.dev_t4_boehlke2001(tensors[key])
            assert np.allclose(D_spencer, D_boehlke)

    def test_deviator_part_of_4_tensor_implementations_spencer_simple(self, tensors):
        for key in ["hooke", "complete"]:
            D_spencer = alternatives.dev_t4_spencer1970(tensors[key])
            D_simple = mechkit.operators.dev_tensor_4th_order_simple(tensors[key])
            assert np.allclose(D_spencer, D_simple)
