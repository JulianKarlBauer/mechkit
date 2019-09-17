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
import pytest

sys.path.append(os.path.join('..'))
import mechkit

np.set_printoptions(
        linewidth=140,
        precision=2,
        )


def steel_scalars():
    return {
        'E':    2e5,
        'nu':   0.3,
        'K':    5e5/3.,
        'la':   1.5e6/13,
        'G':    1e6/13,
        'M':    3.5e6/13,
        }


def add_stiffnesses(inp):
    con = mechkit.notation.VoigtConverter(silent=True)
    tensors = mechkit.tensors.Basic()

    inp['stiffness'] = 3.*inp['K']*tensors.P1 + 2.*inp['G']*tensors.P2
    inp['stiffness_mandel6'] = con.to_mandel6(inp['stiffness'])
    inp['stiffness_voigt'] = con.mandel6_to_voigt(
                                    inp['stiffness_mandel6'],
                                    voigt_type='stiffness',
                                    )
    return inp


def add_compliances(inp):
    con = mechkit.notation.VoigtConverter(silent=True)

    inp['compliance_mandel6'] = np.linalg.inv(inp['stiffness_mandel6'])
    inp['compliance'] = con.to_tensor(inp['compliance_mandel6'])
    inp['compliance_voigt'] = con.mandel6_to_voigt(
                                    inp['compliance_mandel6'],
                                    voigt_type='compliance',
                                    )


def add_tensors(inp):
    out = copy.deepcopy(inp)
    add_stiffnesses(out)
    add_compliances(out)
    return out


def add_additive_tensors(inp):
    out = copy.deepcopy(inp)
    out.pop('nu')
    add_stiffnesses(out)
    return out


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_variants_arguments_steel():
    steel = steel_scalars()
    steel_tensor = add_tensors(inp=steel)
    for comb in itertools.combinations(steel.keys(), 2):
        mat = mechkit.material.Isotropic(**{k: steel[k] for k in comb})
        for key, val in steel_tensor.items():
            assert np.allclose(getattr(mat, key), val)


def test_reference():
    return {
            'C11_voigt': 15.755,                  # e11 Pa
            'C44_voigt': 5.3184,                  # e11 Pa
            'C12_voigt': 5.11850,                 # e11 Pa
            'nu': 0.2452,
            'E': 13.245,                    # e11 Pa
            }


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_reference_values():
    inp = test_reference()
    for comb in itertools.combinations(inp.keys(), 2):
        print('######################\n', comb)
        mat = mechkit.material.Isotropic(**{k: inp[k] for k in comb})
        keys = ['E', 'nu']
        for key in keys:
            calculated = getattr(mat, key)
            reference = inp[key]
            print(key)
            print('calculated: ', calculated)
            print('reference: ', reference)
            print()
            assert np.allclose(
                        calculated,
                        reference,
                        rtol=1e-4,
                        atol=1e-3,
                        )


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_access_dict_like():
    steel = steel_scalars()
    steel_tensor = add_tensors(inp=steel)
    for comb in itertools.combinations(steel.keys(), 2):
        mat = mechkit.material.Isotropic(**{k: steel[k] for k in comb})
        for key, val in steel_tensor.items():
            assert np.allclose(mat[key], val)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_use_aliases():
    # Get aliases
    mat = mechkit.material.Isotropic(E=1, nu=0.3)
    aliases = mat._get_names_aliases()

    steel = steel_scalars()
    steel_tensor = add_tensors(inp=steel)

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

    inp = steel_scalars()
    result = add_tensors(inp)
    additive_result = add_additive_tensors(inp)

    comb = ['K', 'G']
    mat0 = mechkit.material.Isotropic(**{k: additive_result[k] for k in comb})
    mat = mat0 + mat0

    for key, val in additive_result.items():
        assert np.allclose(mat[key], 2*val)

    for key in ['compliance_mandel6', 'compliance']:
        assert np.allclose(mat[key], 0.5*result[key])


def test_arithmetic_mult_sub():

    inp = steel_scalars()
    additive_result = add_additive_tensors(inp)

    comb = ['K', 'G']
    mat0 = mechkit.material.Isotropic(**{k: additive_result[k] for k in comb})
    mat = 4*mat0 - mat0

    for key, val in additive_result.items():
        assert np.allclose(mat[key], 3*val)


def test_exception_nbr_parameter():
    with pytest.raises(mechkit.utils.Ex) as excinfo:
        mechkit.material.Isotropic(E=10, G=15, nu=0.3)
    assert "Number of" in str(excinfo.value)


def test_exception_duplicate_parameter():
    with pytest.raises(mechkit.utils.Ex) as excinfo:
        mechkit.material.Isotropic(E=10, youngs_modulus=15)
    assert "Redundant" in str(excinfo.value)


if __name__ == '__main__':
    test_reference_values()




