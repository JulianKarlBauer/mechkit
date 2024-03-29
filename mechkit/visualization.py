#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mechkit


class StiffnessAnalyser(object):
    """Analyse linear elastic stiffness.
    Vectorized calculation of Youngs- and generalized compression modulus
    following :cite:p:`Boehlke2001`

    Examples
    --------
    >>> import mechkit
    >>> import numpy as np

    >>> C = np.array(
         [[18755,  6444,  4666,     0,     0,  3754],
          [6444,  9565,  4933,     0,     0,   840],
          [4666,  4933,  8665,     0,     0,  -133],
          [   0,     0,     0,  1844,    35,     0],
          [   0,     0,     0,    35,  1915,     0],
          [3754,   840,  -133,     0,     0,  3858]])
    >>> v = mechkit.visualization.StiffnessAnalyser(C)
    >>> v.E_in_direction([1, 0, 0])
    """

    def __init__(self, stiffness):

        self.notation = "tensor"
        self.con = mechkit.notation.Converter()
        tensors = mechkit.tensors.Basic()
        self.P1 = tensors.P1
        self.P2 = tensors.P2
        self.I2 = tensors.I2

        self.stiffness = self.con.to_tensor(stiffness)

    @property
    def E_RI(self):
        return 1.0 / (2.0 / (3.0 * self._h2) + 1.0 / (3.0 * self._h1))

    @property
    def K_RI(self):
        return self._h1 / 3.0

    @property
    def _h1(self):
        """norm(P1) = 1"""
        return np.einsum("ijkl, ijkl->", self.stiffness, self.P1)

    @property
    def _h2(self):
        return (
            np.einsum("ijkl, ijkl->", self.stiffness, self.P2)
            / np.linalg.norm(self.P2) ** 2
        )

    @property
    def _compliance(self):
        return self.con.to_tensor(np.linalg.inv(self.con.to_mandel6(self.stiffness)))

    def _assert_direction_three_dimensional(self, direction):
        """
        Assert last dimension is three-dimensional vector dimension
        """
        assert np.array(direction).shape[-1] == 3

    def _normalize_direction(self, direction):
        return direction / np.linalg.norm(direction, axis=-1)[..., np.newaxis]

    def E_in_direction(self, direction, normalize=False):
        """
        Calculate Youngs modulus in specified direction
        """

        self._assert_direction_three_dimensional(direction=direction)
        # Normalize direction vector(s)
        d = self._normalize_direction(direction=direction)

        S = self._compliance
        E = 1.0 / np.einsum("...i, ...j, ijkl, ...k, ...l -> ...", d, d, S, d, d)

        if normalize:
            E = E / self.E_RI
        return E

    def K_in_direction(self, direction, normalize=False):
        """
        Calculate generalized compression modulus in specified direction

        Generalized compression modulus represents the change of volume due
        to uniaxial tension in the specified direction.
        """

        self._assert_direction_three_dimensional(direction=direction)
        # Normalize direction vector(s)
        d = self._normalize_direction(direction=direction)

        S = self._compliance
        K = 1.0 / (3.0 * np.einsum("ij, ijkl, ...k, ...l -> ...", self.I2, S, d, d))

        if normalize:
            K = K / self.K_RI
        return K
