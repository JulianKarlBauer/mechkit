#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mechkit


class Test_StiffnessAnalyser:
    def test_get_E_in_principal_direction_and_perpendicular_transverselly_iso(self):

        direction = [1, 1, 0]
        E_l = 100.0
        E_t = 20.0

        mat = mechkit.material.TransversalIsotropic(
            E_l=E_l,
            E_t=E_t,
            nu_lt=0.3,
            G_lt=10.0,
            G_tt=7.0,
            principal_axis=direction,
        )
        analyzer = mechkit.visualization.StiffnessAnalyser(stiffness=mat.stiffness)

        # E Modulus in principal direction has to be equal to E_l
        assert np.allclose(analyzer.E_in_direction(direction=direction), E_l)

        # E Modulus in any direction perpendicular to principal direction
        # has to be equal to E_t
        assert np.allclose(analyzer.E_in_direction(direction=[0, 0, 1]), E_t)

    def test_E_and_K_generalized_of_isotropic_vectorized(self):
        E_modul = 2e3
        K_modul = 1e3

        mat = mechkit.material.Isotropic(E=E_modul, K=K_modul)
        analyzer = mechkit.visualization.StiffnessAnalyser(stiffness=mat.stiffness)

        shape = (2, 4)
        tmp_1, tmp_2 = shape

        # Unpacking by "*shape" is not valid Python2.x
        directions = np.random.rand(tmp_1, tmp_2, 3)

        youngs_moduli = analyzer.E_in_direction(direction=directions)
        print(youngs_moduli)
        assert youngs_moduli.shape == shape
        assert np.allclose(youngs_moduli, np.ones(shape) * E_modul)

        gen_bulk_modulus = analyzer.K_in_direction(direction=directions)
        print(gen_bulk_modulus)
        assert gen_bulk_modulus.shape == shape
        assert np.allclose(gen_bulk_modulus, np.ones(shape) * K_modul)


if __name__ == "__main__":
    instance = Test_StiffnessAnalyser()
    instance.test_get_E_in_principal_direction_and_perpendicular_transverselly_iso()
    instance.test_E_and_K_generalized_of_isotropic_vectorized()
