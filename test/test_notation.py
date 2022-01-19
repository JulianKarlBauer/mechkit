#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run tests:
    python3 -m pytest
"""

import numpy as np
import pytest
import mechkit

##################################
# Helpers


def assertException(func, message, args=[], kwargs={}, exception=mechkit.utils.Ex):
    with pytest.raises(exception) as excinfo:
        func(*args, **kwargs)
    assert str(excinfo.value).startswith(message)
    return None


##################################
# Test


class Test_Converter:
    def test_unsupported_shape(self):

        con = mechkit.notation.Converter()
        assertException(
            con.to_mandel6,
            "Tensor shape not supported",
            args=[],
            kwargs={"inp": np.ones((3, 2))},
            exception=mechkit.utils.Ex,
        )

    def test_compare_P1_P2_mandel6_tensor(self):

        con = mechkit.notation.Converter()
        t = mechkit.tensors.Basic()

        # Prepare
        P1_mandel6 = (
            1.0
            / 3.0
            * np.array(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float64",
            )
        )

        I4s_mandel6 = np.eye(6, dtype="float64")

        P2_mandel6 = I4s_mandel6 - P1_mandel6

        # t4_to_mandel
        assert np.allclose(P1_mandel6, con.to_mandel6(t.P1))
        assert np.allclose(P2_mandel6, con.to_mandel6(t.P2))
        assert np.allclose(I4s_mandel6, con.to_mandel6(t.I4s))

        # mandel2_to_tensor
        assert np.allclose(t.P1, con.to_tensor(P1_mandel6))
        assert np.allclose(t.P2, con.to_tensor(P2_mandel6))
        assert np.allclose(t.I4s, con.to_tensor(I4s_mandel6))

    def test_level2_mandel6_tensor(self):

        con = mechkit.notation.Converter()

        mandel = np.array(
            [1.0, 2.0, 3.0, np.sqrt(2) * 4.0, np.sqrt(2) * 5.0, np.sqrt(2) * 6.0],
            dtype="float64",
        )

        tensor = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])

        assert np.allclose(con.to_mandel6(tensor), mandel)
        assert np.allclose(con.to_tensor(mandel), tensor)

    def test_level4_mandel6_tensor(self):

        con = mechkit.notation.Converter()
        factor = np.sqrt(2.0)
        mandel = np.array(
            [
                [1.0, 1.0, 1.0, factor, factor, factor],
                [1.0, 1.0, 1.0, factor, factor, factor],
                [1.0, 1.0, 1.0, factor, factor, factor],
                [factor, factor, factor, 2.0, 2.0, 2.0],
                [factor, factor, factor, 2.0, 2.0, 2.0],
                [factor, factor, factor, 2.0, 2.0, 2.0],
            ],
            dtype="float64",
        )

        tensor = np.ones((3, 3, 3, 3))

        assert np.allclose(con.to_mandel6(tensor), mandel)
        assert np.allclose(con.to_tensor(mandel), tensor)

    def test_pass_throught(self):

        con = mechkit.notation.Converter()

        m6_2 = np.random.rand(6)
        m6_4 = np.random.rand(6, 6)
        m9_2 = np.random.rand(9)
        m9_4 = np.random.rand(9, 9)
        t2 = np.random.rand(3, 3)
        t4 = np.random.rand(3, 3, 3, 3)

        assert np.allclose(con.to_mandel6(m6_2), m6_2)
        assert np.allclose(con.to_mandel6(m6_4), m6_4)
        assert np.allclose(con.to_mandel9(m9_2), m9_2)
        assert np.allclose(con.to_mandel9(m9_4), m9_4)
        assert np.allclose(con.to_tensor(t2), t2)
        assert np.allclose(con.to_tensor(t4), t4)

    def test_mandel6_to_tensor_to_mandel6(self):

        con = mechkit.notation.Converter()
        matrix = np.random.rand(6, 6)
        assert np.allclose(con.to_mandel6(con.to_tensor(matrix)), matrix)

    def test_tensor_to_mandel6_to_tensor(self):

        con = mechkit.notation.Converter()

        tensor = np.random.rand(3, 3, 3, 3)
        tensor_sym_minor = 0.25 * (
            tensor.transpose([0, 1, 2, 3])
            + tensor.transpose([1, 0, 2, 3])
            + tensor.transpose([0, 1, 3, 2])
            + tensor.transpose([1, 0, 3, 2])
        )
        matrix = con.to_mandel6(tensor_sym_minor)

        assert np.allclose(con.to_tensor(matrix), tensor_sym_minor)

    def test_mandel9_to_tensor_to_mandel9(self):

        con = mechkit.notation.Converter()
        matrix = np.random.rand(9, 9)
        assert np.allclose(con.to_mandel9(con.to_tensor(matrix)), matrix)

    def test_tensor_to_mandel9_to_tensor(self):

        con = mechkit.notation.Converter()
        tensor = np.random.rand(3, 3, 3, 3)
        assert np.allclose(con.to_tensor(con.to_mandel9(tensor)), tensor)

    def test_ones_tensors_to_mandel6_to_voigt_to_mandel6(self):
        """Define ones tensors and transform to Mandel.

        Ones tensors are useful to visualize the conversions.
        Ones tensors are not useful to check correct implementation!
        Convert this mandel representation to Voigt and back
        to mandel and compare with initial mandel representation"""

        converter = mechkit.notation.VoigtConverter()

        ones2_mandel = converter.to_mandel6(np.ones((3, 3)))
        ones4_mandel = converter.to_mandel6(np.ones((3, 3, 3, 3)))

        voigt_types = {
            "stress": ones2_mandel,
            "strain": ones2_mandel,
            "stiffness": ones4_mandel,
            "compliance": ones4_mandel,
        }

        print("#####################")
        print("Input in Mandel")
        for voigt_type, inp in voigt_types.items():
            print(voigt_type)
            print(inp)

        print("#####################")
        print("In Voigt")
        voigts = {}

        for voigt_type, input_mandel in voigt_types.items():
            out = converter.mandel6_to_voigt(inp=input_mandel, voigt_type=voigt_type)
            print(voigt_type)
            print(out)

            voigts[voigt_type] = out

        print("#####################")
        print("Back in Mandel")
        mandels = {}

        for voigt_type, voigt in voigts.items():
            out = converter.voigt_to_mandel6(inp=voigt, voigt_type=voigt_type)
            print(voigt_type)
            print(out)

            mandels[voigt_type] = out

        for voigt_type, mandel in mandels.items():
            assert np.allclose(mandel, voigt_types[voigt_type])

    def test_to_like(
        self,
    ):
        con = mechkit.notation.Converter()

        m6_2 = np.random.rand(6)
        m6_4 = np.random.rand(6, 6)
        m9_2 = np.random.rand(9)
        m9_4 = np.random.rand(9, 9)
        t2 = np.random.rand(3, 3)
        t4 = np.random.rand(3, 3, 3, 3)

        t2_sym = con.to_tensor(con.to_mandel6(t2))
        t4_sym = con.to_tensor(con.to_mandel6(t4))

        funcs_pairs = {
            con.to_tensor: [
                {"inp": t2, "like": t2},
                {"inp": m6_2, "like": t2},
                {"inp": m9_2, "like": t2},
                {"inp": t4, "like": t4},
                {"inp": m6_4, "like": t4},
                {"inp": m9_4, "like": t4},
            ],
            con.to_mandel6: [
                {"inp": t2_sym, "like": m6_2},
                {"inp": m6_2, "like": m6_2},
                {"inp": m9_2, "like": m6_2},
                {"inp": t4_sym, "like": m6_4},
                {"inp": m6_4, "like": m6_4},
                {"inp": m9_4, "like": m6_4},
            ],
            con.to_mandel9: [
                {"inp": t2, "like": m9_2},
                {"inp": m6_2, "like": m9_2},
                {"inp": m9_2, "like": m9_2},
                {"inp": t4, "like": m9_4},
                {"inp": m6_4, "like": m9_4},
                {"inp": m9_4, "like": m9_4},
            ],
        }

        for func, pairs in funcs_pairs.items():
            for pair in pairs:
                inp = pair["inp"]
                like = pair["like"]

                assert con.to_like(inp=inp, like=like).shape == like.shape

                assert np.allclose(con.to_like(inp=inp, like=like), func(inp))

    ##################################
    # Test eigenvalues

    def isotropic_stiffness_mandel6(self, EW1, EW2):
        con = mechkit.notation.Converter()
        tensors = mechkit.tensors.Basic()
        P1 = con.to_mandel6(tensors.P1)
        P2 = con.to_mandel6(tensors.P2)
        return P1 * EW1 + P2 * EW2

    def compare_matrix_eigenvalues_with_list_of_numbers(
        self, matrix, list_of_numbers, decimals=7
    ):
        boolean = set(np.linalg.eig(matrix)[0].round(decimals=decimals)) == set(
            np.array(list_of_numbers).round(decimals=decimals)
        )
        return boolean

    def test_eigenvalues_of_isotropic_stiffness_mandel6(self):
        EW1 = 1500
        EW2 = 700

        C = self.isotropic_stiffness_mandel6(EW1, EW2)

        assert self.compare_matrix_eigenvalues_with_list_of_numbers(
            matrix=C, list_of_numbers=[EW1, EW2]
        )

    def test_eigenvalues_of_inverse_of_isotropic_stiffness_mandel6(self):
        EW1 = 1500
        EW2 = 700

        C = self.isotropic_stiffness_mandel6(EW1, EW2)

        C_inv = np.linalg.inv(C)

        assert self.compare_matrix_eigenvalues_with_list_of_numbers(
            matrix=C_inv, list_of_numbers=[1.0 / EW1, 1.0 / EW2]
        )


@pytest.fixture
def con_aba():
    return mechkit.notation.AbaqusConverter(silent=True)


class Test_UmatConverter:
    def test_umat_stress(self, con):

        mandel = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        umat = con.convert(
            inp=mandel,
            source="mandel6",
            target="umat",
            quantity="stress",
        )

        print("Umat")
        print(umat)
        fac = 1.0 / np.sqrt(2)
        assert np.allclose(
            umat, np.array([1.0, 2.0, 3.0, 6.0 * fac, 5 * fac, 4.0 * fac])
        )

    def test_umat_stiffness(self, con):

        tensor = np.arange(81).reshape(3, 3, 3, 3)
        # Attention: Discard of non-symmetric parts is intended

        umat = con.convert(
            inp=tensor,
            source="tensor",
            target="umat",
            quantity="stiffness",
        )

        print("Umat")
        print(umat)
        assert np.allclose(
            umat,
            np.array(
                [
                    [0.0, 4.0, 8.0, 2.0, 4.0, 6.0],
                    [36.0, 40.0, 44.0, 38.0, 40.0, 42.0],
                    [72.0, 76.0, 80.0, 74.0, 76.0, 78.0],
                    [18.0, 22.0, 26.0, 20.0, 22.0, 24.0],
                    [36.0, 40.0, 44.0, 38.0, 40.0, 42.0],
                    [54.0, 58.0, 62.0, 56.0, 58.0, 60.0],
                ]
            ),
        )


@pytest.fixture(name="tensor_min_sym")
def create_random_tensors_with_minor_symmetries(shape_vectorized=(1,)):
    shapes_mandel6 = {
        "stress": shape_vectorized + (6,),
        "strain": shape_vectorized + (6,),
        "stiffness": shape_vectorized + (6, 6),
        "compliance": shape_vectorized + (6, 6),
    }

    tensors = {key: np.random.rand(*shape) for key, shape in shapes_mandel6.items()}
    return tensors


@pytest.fixture(name="tensor_no_sym")
def create_random_tensors_without_symmetry(shape_vectorized=(1,)):
    shapes = {
        "stress": shape_vectorized + (3, 3),
        "strain": shape_vectorized + (3, 3),
        "stiffness": shape_vectorized + (3, 3, 3, 3),
        "compliance": shape_vectorized + (3, 3, 3, 3),
    }

    tensors = {key: np.random.rand(*shape) for key, shape in shapes.items()}
    return tensors


@pytest.fixture(name="con")
def explicit_converter():
    return mechkit.notation.ExplicitConverter()


class Test_ExplicitConverter:
    def test_loop_minor_sym(self, con, tensor_min_sym):
        excluded_notations = ["abaqusMaterialAnisotropic"]
        start_notation = "mandel6"
        for key_quantity, graph in con.graphs_dict.items():
            nodes = graph.nodes
            nodes_without_start = [
                node
                for node in nodes
                if ((node != start_notation) and (node not in excluded_notations))
            ]
            for target in nodes_without_start:
                origin = tensor_min_sym[key_quantity]
                new = con.convert(
                    inp=origin,
                    source=start_notation,
                    target=target,
                    quantity=key_quantity,
                )
                back = con.convert(
                    inp=new,
                    source=target,
                    target=start_notation,
                    quantity=key_quantity,
                )
                print("\n\n\n {}: {}".format(key_quantity, target))
                print(origin)
                print(back)
                print(new)
                assert np.allclose(origin, back)

    def test_loop_no_sym(self, con, tensor_no_sym):
        nodes = ["mandel9"]
        start_notation = "tensor"

        for key_quantity, graph in con.graphs_dict.items():
            for target in nodes:
                origin = tensor_no_sym[key_quantity]
                new = con.convert(
                    inp=origin,
                    source=start_notation,
                    target=target,
                    quantity=key_quantity,
                )
                back = con.convert(
                    inp=new,
                    source=target,
                    target=start_notation,
                    quantity=key_quantity,
                )
                print("\n\n\n {}: {}".format(key_quantity, target))
                print(origin)
                print(back)
                print(new)
                assert np.allclose(origin, back)

    def test_loop_inner_sym_stiffness(self, con):
        voigt = np.random.rand(6, 6)
        origin = voigt_inner_sym = voigt + voigt.T
        print(voigt)
        print(voigt_inner_sym)
        new = con.convert(
            inp=origin,
            source="voigt",
            target="abaqusMaterialAnisotropic",
            quantity="stiffness",
        )
        back = con.convert(
            inp=new,
            source="abaqusMaterialAnisotropic",
            target="voigt",
            quantity="stiffness",
        )
        print(origin)
        print(back)
        print(new)
        assert np.allclose(origin, back)


if __name__ == "__main__":
    pass
