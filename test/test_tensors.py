#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Run tests:
    python3 -m pytest
'''

import os
import sys
import numpy as np
import pprint

sys.path.append(os.path.join('..'))
import mechkit

basic = mechkit.tensors.Basic()
con = mechkit.notation.Converter()


##########################################
# Helpers

def random_stiffness():
    '''Create random tensor of fourth order with inner symmetry,
    i.e. minor and major symmetry
    '''
    tensor = np.random.rand(3, 3, 3, 3,)
    t_mandel = con.to_mandel6(tensor)
    return con.to_tensor(0.5 * (t_mandel + t_mandel.transpose()))


def define_K_G_in_stiffness(stiffness, K, G):
    '''Replace isotropic part by known isotropic stiffness
    '''
    s = con.to_mandel6(stiffness)
    m = mechkit.material.Isotropic(K=K, G=G)

    for r, row in enumerate(m.stiffness_mandel6):
        for c, col in enumerate(row):
            if not np.allclose(col, 0.):
                s[r, c] = col
    return con.to_tensor(s)

##########################################
# Tests


def test_I2():
    it = np.array(
            [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ]
            )
    assert np.allclose(basic.I2, it)


def test_I4():
    it = np.array(
            [
                [1., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 1., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 1., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 1., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 1., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., ],
            ]
            )
    assert np.allclose(con.to_mandel9(basic.I4), it)


def test_I4s():
    it = np.array(
            [
                [1., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 1., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 1., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 1., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 1., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., ],
            ]
            )
    assert np.allclose(con.to_mandel9(basic.I4s), it)


def test_I4a():
    it = np.array(
            [
                [0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 1., 0., ],
                [0., 0., 0., 0., 0., 0., 0., 0., 1., ],
            ]
            )
    assert np.allclose(con.to_mandel9(basic.I4a), it)


def test_biorthogonal_P1P2():
    a = np.einsum('ijkl, klmn ->ijmn', basic.P1, basic.P2)
    assert np.allclose(a, np.zeros((a.shape)))


def test_biorthogonal_P2P1():
    a = np.einsum('ijkl, klmn ->ijmn', basic.P2, basic.P1)
    assert np.allclose(a, np.zeros((a.shape)))


def test_idempotenz_P1():
    a = np.einsum('ijkl, klmn ->ijmn', basic.P1, basic.P1)
    assert np.allclose(a, basic.P1)


def test_idempotenz_P2():
    a = np.einsum('ijkl, klmn ->ijmn', basic.P2, basic.P2)
    assert np.allclose(a, basic.P2)


def test_norm_P1():
    assert np.allclose(np.linalg.norm(basic.P1), 1.)


def test_norm_P2():
    assert np.allclose(np.linalg.norm(basic.P2), np.sqrt(5))


def test_effect_P1():
    K = 100
    G = 61
    rs = random_stiffness()
    s = define_K_G_in_stiffness(rs, K, G)

    K_result = np.einsum('ijkl, ijkl ->', s, basic.P1) / 3.
    assert np.allclose(K, K_result)


def test_effect_P2():
    '''Norm of
    '''
    K = 145
    G = 43
    rs = random_stiffness()
    s = define_K_G_in_stiffness(rs, K, G)

    G_result = np.einsum('ijkl, ijkl ->', s, basic.P2) / 2. / 5.
    assert np.allclose(G, G_result)


def test_I6s():
    # Note:
    # Does A have to be symmetric for formula of JA being valid?
    # Or is the implementation of I6s wrong?
    A = np.einsum('ijkl, kl-> ij', basic.I4s, np.random.rand(3, 3))
    I2 = basic.I2
    JA = 1./4. * (
                np.einsum('im, jn ->ijmn', A, I2) +
                np.einsum('in, jm ->ijmn', A, I2) +
                np.einsum('jn, im ->ijmn', A, I2) +
                np.einsum('jm, in ->ijmn', A, I2)
                )
    I6sA = np.einsum('ijklmn, mn->ijkl', basic.I6s, A)
    pprint.pprint(con.to_mandel9(JA))
    pprint.pprint(con.to_mandel9(I6sA))
    assert np.allclose(JA, I6sA)


if __name__ == '__main__':
    rs = random_stiffness()
    s = define_K_G_in_stiffness(stiffness=rs)
