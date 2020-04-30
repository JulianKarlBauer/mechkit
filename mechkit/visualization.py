#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mechkit


class StiffnessAnalyser(object):
    '''Analyse linear elastic stiffness

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
    '''

    def __init__(self, stiffness):

        self.notation = 'tensor'
        self.con = mechkit.notation.Converter()
        tensors = mechkit.tensors.Basic()
        self.P1 = tensors.P1
        self.P2 = tensors.P2
        self.I2 = tensors.I2

        self.stiffness = self.con.to_tensor(stiffness)

    @property
    def E_RI(self, ):
        return 1./(2./(3.*self._h2) + 1./(3.*self._h1))

    @property
    def K_RI(self, ):
        return self._h1 / 3.

    @property
    def _h1(self, ):
        '''norm(P1) = 1'''
        return np.einsum('ijkl, ijkl->', self.stiffness, self.P1)

    @property
    def _h2(self, ):
        return np.einsum('ijkl, ijkl->', self.stiffness, self.P2) \
             / np.linalg.norm(self.P2)**2

    @property
    def _compliance(self, ):
        return self.con.to_tensor(
                            np.linalg.inv(
                                self.con.to_mandel6(self.stiffness)
                                )
                            )

    def E_in_direction(self, direction, normalize=False):
        '''
        Calculate Youngs modulus in specified direction
        '''
        d = direction / np.linalg.norm(direction)
        S = self._compliance
        E = 1. / np.einsum('i, j, ijkl, k, l -> ', d, d, S, d, d)

        if normalize:
            E = E / self.E_RI
        return E

    def K_in_direction(self, direction, normalize=False):
        '''
        Calculate generalized compression modulus in specified direction

        Generalized compression modulus represents the change of volume due
        to uniaxial tension in the specified direction.
        '''

        d = direction / np.linalg.norm(direction)
        S = self._compliance
        I2 = mechkit.tensors.Basic().I2
        K = 1. / (3. * np.einsum('ij, ijkl, k, l -> ', I2, S, d, d, ))

        if normalize:
            K = K / self.K_RI
        return K


if __name__ == '__main__':

    C = np.array(
         [[18755,  6444,  4666,     0,     0,  3754],
          [6444,  9565,  4933,     0,     0,   840],
          [4666,  4933,  8665,     0,     0,  -133],
          [   0,     0,     0,  1844,    35,     0],
          [   0,     0,     0,    35,  1915,     0],
          [3754,   840,  -133,     0,     0,  3858]])
    v = mechkit.visualization.StiffnessAnalyser(C)
    v.E_in_direction([1, 0, 0])


