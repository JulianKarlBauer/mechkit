#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''Run tests:
    python3 -m pytest
'''

import sys
import os
import numpy as np
import itertools
import copy

sys.path.append(os.path.join('..'))
import mechkit

np.set_printoptions(
        linewidth=140,
        precision=2,
        )


def get_steel_scalar():
    return {
        'E':    2e5,
        'nu':   0.3,
        'K':    5e5/3.,
        'la':   1.5e6/13,
        'G':    1e6/13,
        }


def get_steel_tensor():
    steel = get_steel_scalar()

    con = mechkit.notation.Converter()
    tensors = mechkit.tensors.Basic()

    steel['stiffness'] = 3.*steel['K']*tensors.P1 + 2.*steel['G']*tensors.P2
    steel['stiffness_mandel6'] = con.to_mandel6(steel['stiffness'])
    steel['compliance_mandel6'] = np.linalg.inv(steel['stiffness_mandel6'])
    steel['compliance'] = con.to_tensor(steel['compliance_mandel6'])
    return steel


def get_additive_steel_tensor():
    steel = get_steel_tensor()
    steel.pop('nu')
    steel.pop('compliance_mandel6')
    steel.pop('compliance')
    return steel


def test_variants_arguments_steel():
    steel = get_steel_scalar()
    steel_tensor = get_steel_tensor()
    for comb in itertools.combinations(steel.keys(), 2):
        mat = mechkit.material.Isotropic(**{k: steel[k] for k in comb})
        for key, val in steel_tensor.items():
            assert np.allclose(getattr(mat, key), val)


def test_access_dict_like():
    steel = get_steel_scalar()
    steel_tensor = get_steel_tensor()
    for comb in itertools.combinations(steel.keys(), 2):
        mat = mechkit.material.Isotropic(**{k: steel[k] for k in comb})
        for key, val in steel_tensor.items():
            assert np.allclose(mat[key], val)


def test_use_aliases():
    # Get aliases
    mat = mechkit.material.Isotropic(E=1, nu=0.3)
    aliases = mat._get_names_aliases()

    steel = get_steel_scalar()
    steel_tensor = get_steel_tensor()

    for comb in itertools.combinations(steel.keys(), 2):
        for alias0 in aliases[comb[0]]:
            for alias1 in aliases[comb[1]]:
                mat = mechkit.material.Isotropic(
                                **{
                                    alias0:     steel[comb[0]],
                                    alias1:     steel[comb[1]],
                                    }
                                )
                for key, val in steel_tensor.items():
                    assert np.allclose(getattr(mat, key), val)


def test_arithmetic_add():
    additive_steel = get_additive_steel_tensor()

    comb = ['K', 'G']
    mat0 = mechkit.material.Isotropic(**{k: additive_steel[k] for k in comb})
    mat = mat0 + mat0

    for key, val in additive_steel.items():
        assert np.allclose(mat[key], 2*val)

    for key in ['compliance_mandel6', 'compliance']:
        val = get_steel_tensor()[key]
        assert np.allclose(mat[key], 0.5*val)


def test_arithmetic_mult_sub():
    additive_steel = get_additive_steel_tensor()

    comb = ['K', 'G']
    mat0 = mechkit.material.Isotropic(**{k: additive_steel[k] for k in comb})
    mat = 4*mat0 - mat0

    for key, val in additive_steel.items():
        assert np.allclose(mat[key], 3*val)


if __name__ == '__main__':
    test_arithmetic_add()




