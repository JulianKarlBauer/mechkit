#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Run tests:
    python3 -m pytest
"""

import sys
import os
import numpy as np
import itertools
import copy
import pytest

sys.path.append(os.path.join(".."))
import mechkit

np.set_printoptions(
    linewidth=140, precision=2,
)

##############################################################################
# Isotropic


def steel_scalars():
    return {
        "E": 2e5,
        "nu": 0.3,
        "K": 5e5 / 3.0,
        "la": 1.5e6 / 13,
        "G": 1e6 / 13,
        "M": 3.5e6 / 13,
    }


def add_stiffnesses(inp):
    con = mechkit.notation.VoigtConverter(silent=True)
    tensors = mechkit.tensors.Basic()

    inp["stiffness"] = 3.0 * inp["K"] * tensors.P1 + 2.0 * inp["G"] * tensors.P2
    inp["stiffness_mandel6"] = con.to_mandel6(inp["stiffness"])
    inp["stiffness_voigt"] = con.mandel6_to_voigt(
        inp["stiffness_mandel6"], voigt_type="stiffness",
    )
    return inp


def add_compliances(inp):
    con = mechkit.notation.VoigtConverter(silent=True)

    inp["compliance_mandel6"] = np.linalg.inv(inp["stiffness_mandel6"])
    inp["compliance"] = con.to_tensor(inp["compliance_mandel6"])
    inp["compliance_voigt"] = con.mandel6_to_voigt(
        inp["compliance_mandel6"], voigt_type="compliance",
    )


def add_tensors(inp):
    out = copy.deepcopy(inp)
    add_stiffnesses(out)
    add_compliances(out)
    return out


def add_additive_tensors(inp):
    out = copy.deepcopy(inp)
    out.pop("nu")
    add_stiffnesses(out)
    return out


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_variants_arguments_steel():
    steel = steel_scalars()
    steel_tensor = add_tensors(inp=steel)
    for comb in itertools.combinations(steel.keys(), 2):
        mat = mechkit.material.Isotropic(**{k: steel[k] for k in comb})
        for key, val in steel_tensor.items():
            assert np.allclose(getattr(mat, key), val)


def test_reference():
    return {
        "C11_voigt": 15.755,  # e11 Pa
        "C44_voigt": 5.3184,  # e11 Pa
        "C12_voigt": 5.11850,  # e11 Pa
        "nu": 0.2452,
        "E": 13.245,  # e11 Pa
    }


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_reference_values():
    inp = test_reference()
    for comb in itertools.combinations(inp.keys(), 2):
        print("######################\n", comb)
        mat = mechkit.material.Isotropic(**{k: inp[k] for k in comb})
        keys = ["E", "nu"]
        for key in keys:
            calculated = getattr(mat, key)
            reference = inp[key]
            print(key)
            print("calculated: ", calculated)
            print("reference: ", reference)
            print()
            assert np.allclose(calculated, reference, rtol=1e-4, atol=1e-3,)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_access_dict_like():
    steel = steel_scalars()
    steel_tensor = add_tensors(inp=steel)
    for comb in itertools.combinations(steel.keys(), 2):
        mat = mechkit.material.Isotropic(**{k: steel[k] for k in comb})
        for key, val in steel_tensor.items():
            assert np.allclose(mat[key], val)


@pytest.mark.filterwarnings("ignore::UserWarning")
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
                    **{alias0: steel[comb[0]], alias1: steel[comb[1]],}
                )
                for key, val in steel_tensor.items():
                    assert np.allclose(getattr(mat, key), val)


def test_arithmetic_add():

    inp = steel_scalars()
    result = add_tensors(inp)
    additive_result = add_additive_tensors(inp)

    comb = ["K", "G"]
    mat0 = mechkit.material.Isotropic(**{k: additive_result[k] for k in comb})
    mat = mat0 + mat0

    for key, val in additive_result.items():
        assert np.allclose(mat[key], 2 * val)

    for key in ["compliance_mandel6", "compliance"]:
        assert np.allclose(mat[key], 0.5 * result[key])


def test_arithmetic_mult_sub():

    inp = steel_scalars()
    additive_result = add_additive_tensors(inp)

    comb = ["K", "G"]
    mat0 = mechkit.material.Isotropic(**{k: additive_result[k] for k in comb})
    mat = 4 * mat0 - mat0

    for key, val in additive_result.items():
        assert np.allclose(mat[key], 3 * val)


def test_exception_nbr_parameter():
    with pytest.raises(mechkit.utils.Ex) as excinfo:
        mechkit.material.Isotropic(E=10, G=15, nu=0.3)
    assert "Number of" in str(excinfo.value)


def test_exception_duplicate_parameter():
    with pytest.raises(mechkit.utils.Ex) as excinfo:
        mechkit.material.Isotropic(E=10, youngs_modulus=15)
    assert "Redundant" in str(excinfo.value)


##############################################################################
# Transversal-Isotropic


class Test_TransversalIsotropic:
    def test_compare_with_data(self,):
        """Thanks to Tarkes Dora Pallicity for kindly supplying the data"""
        # 3 is the fiber direction
        self.engineering = {
            "E11": 5.3270039971985339,
            "V12": 0.56298804,
            "V13": 0.090007581,
            "E22": 5.32534381451564,
            "V21": 0.56281298,
            "V23": 0.090009078,
            "E33": 20.473530537649701,
            "V31": 0.34592915,
            "V32": 0.34604305,
            "G12": 1.7033673797711393,
            "G13": 1.7748275369398245,
            "G23": 1.7747282490254996,
        }
        # Mandel6
        self.cij = C = {
            "11": 8.8103098279815111,
            "12": 5.401109750542668,
            "13": 4.9167594461656954,
            "21": 5.4011063730662592,
            "22": 8.8076619701439434,
            "23": 4.9162303281442874,
            "31": 4.9167753488207184,
            "32": 4.9162475330973479,
            "33": 23.875619726551143,
            "44": 3.5494564980509993,
            "55": 3.5496550738796486,
            "66": 3.4067347595422786,
        }

        E1 = self.engineering["E33"]
        E2 = self.engineering["E11"]
        G12 = self.engineering["G13"]
        G23 = self.engineering["G12"]
        nu12 = self.engineering["V32"]

        self.m = mechkit.material.TransversalIsotropic(
            E_l=E1,
            E_t=E2,
            G_lt=G12,
            G_tt=G23,
            nu_lt=nu12,
            principal_axis=[0, 0, 1],
        )

        self.stiffness = stiffness = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                stiffness[i, j] = C["{}{}".format(i + 1, j + 1)]
        stiffness[3, 3] = C["44"]
        stiffness[4, 4] = C["55"]
        stiffness[5, 5] = C["66"]

        print(self.m.stiffness_mandel6)
        print(self.stiffness)

        assert np.allclose(self.stiffness, self.m.stiffness_mandel6, atol=1e-1)


if __name__ == "__main__":
    c = Test_TransversalIsotropic()
    r = c.test_compare_with_data()
