#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Run tests:
    python3 -m pytest
'''

import os
import sys
import numpy as np
from pprint import pprint

sys.path.append(os.path.join('..'))
import mechkit

basic = mechkit.tensors.Basic()
con = mechkit.notation.Converter()


##########################################
# Helpers

def evenly_distributed_vectors_on_sphere(nbr_vectors=1000):
    '''
    Define nbr_vectors evenly distributed vectors on a sphere

    Using the golden spiral method kindly provided by
    stackoverflow-user "CR Drost"
    https://stackoverflow.com/a/44164075/8935243
    '''
    from numpy import pi, cos, sin, arccos, arange

    indices = arange(0, nbr_vectors, dtype=float) + 0.5

    phi = arccos(1 - 2*indices/nbr_vectors)
    theta = pi * (1 + 5**0.5) * indices

    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    orientations = np.column_stack((x, y, z))
    return orientations


def evenly_distributed_vectors_on_circle_on_zplane(nbr_vectors=1000):
    '''
    Define nbr_vectors evenly distributed vectors on a sphere

    Using the golden spiral method kindly provided by
    stackoverflow-user "CR Drost"
    https://stackoverflow.com/a/44164075/8935243
    '''

    phi = np.linspace(0, 2.*np.pi, nbr_vectors, endpoint=False)

    x, y, z = np.cos(phi), np.sin(phi), np.zeros_like(phi)
    orientations = np.column_stack((x, y, z))
    return orientations


##########################################
# Tests

def test_isotropic_discrete_N2():
    converter = mechkit.notation.Converter()

    orientations = evenly_distributed_vectors_on_sphere(10000)

    basic = converter.to_tensor(mechkit.fabric_tensors.Basic().N2['iso'])
    discrete = mechkit.fabric_tensors.first_kind_discrete(
                                            order=2,
                                            orientations=orientations,
                                            )
    pprint(basic)
    pprint(discrete)

    assert np.allclose(
                basic,
                discrete,
                rtol=1e-6,
                atol=1e-6,
                )


def test_isotropic_discrete_N4():
    converter = mechkit.notation.Converter()

    orientations = evenly_distributed_vectors_on_sphere(10000)

    basic = converter.to_tensor(
                mechkit.fabric_tensors.Basic().N4['iso'],
                )
    discrete = mechkit.fabric_tensors.first_kind_discrete(
                                            order=4,
                                            orientations=orientations,
                                            )
    pprint(basic)
    pprint(discrete)

    assert np.allclose(
                basic,
                discrete,
                rtol=1e-6,
                atol=1e-6,
                )


def test_planar_isotropic_discrete_N4():
    converter = mechkit.notation.Converter()

    orientations = evenly_distributed_vectors_on_circle_on_zplane(10000)

    basic = converter.to_tensor(
                mechkit.fabric_tensors.Basic().N4['planar_iso_xy'],
                )
    discrete = mechkit.fabric_tensors.first_kind_discrete(
                                            order=4,
                                            orientations=orientations,
                                            )
    pprint('basic')
    pprint(converter.to_mandel6(basic))
    pprint('discrete')
    pprint(converter.to_mandel6(discrete))

    assert np.allclose(
                basic,
                discrete,
                rtol=1e-6,
                atol=1e-6,
                )


def test_fabric_tensor_first_kind_discrete():
    '''Compare einsum-implementation with loop-implementation'''

    orientations = np.random.rand(10, 3)     # Ten random vectors in 3D

    # Normalize orientations
    orientations = [np.array(v) / np.linalg.norm(v) for v in orientations]

    def oT_loops(orientations, order=4):
        N = np.zeros((3, ) * order)
        for p in orientations:
            out = p
            for index in range(order-1):
                out = np.multiply.outer(out, p)
            N[:] = N[:] + out
        N = N / len(orientations)
        return N

    for order in range(1, 6):
        assert np.allclose(
                        mechkit.fabric_tensors.first_kind_discrete(
                                                order=order,
                                                orientations=orientations,
                                                ),
                        oT_loops(
                                                order=order,
                                                orientations=orientations,
                                                )
                        )


def test_fabric_tensor_first_kind_discrete_benchmarks():

    orientations = [
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.],
                    ]

    converter = mechkit.notation.Converter()

    f = mechkit.fabric_tensors.first_kind_discrete

    assert np.allclose(
                converter.to_tensor(mechkit.fabric_tensors.Basic().N2['iso']),
                f(
                    order=2,
                    orientations=orientations,
                    ),
                )


if __name__ == '__main__':
    pass
