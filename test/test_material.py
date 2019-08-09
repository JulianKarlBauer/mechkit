#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''Run tests:
    python3 -m pytest
'''

import sys
import os
import numpy as np
import itertools

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


def test_variants_arguments_steel():
    steel = get_steel_scalar()
    steel_tensor = get_steel_tensor()
    for comb in itertools.combinations(steel.keys(), 2):
        mat = mechkit.material.Isotropic(**{k: steel[k] for k in comb})
        for key, val in steel_tensor.items():
            assert np.allclose(getattr(mat, key), val)


def test_use_aliases():
    # Get aliases
    mat = mechkit.material.Isotropic(E=1, nu=0.3)
    aliases = mat._names_aliases

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


if __name__ == '__main__':
    test_use_aliases()




