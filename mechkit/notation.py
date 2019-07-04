#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Notation
'''

import numpy as np


class Converter(object):

    def __init__(self, dtype='float64'):
        '''...'''

        self.dtype = dtype
        self.factor = np.sqrt(2.) / 2.

        self.DIM = 3
        self.DIM_MANDEL6 = 6
        self.DIM_MANDEL9 = 9
        self.BASE6 = self.get_mandel_base_sym()
        self.BASE9 = self.get_mandel_base_skw()

    def get_mandel_base_sym(self,):
        '''Calc orthonormal base for symmetric second order tensors following
        [1] referencing [2], [3]

        This base can be used to transform
            symmetric tensors of second order into vectors with 6 components
            tensors of fourth order with minor symmetries into (6 x 6) matrices

        [1] = Böhlke, T., Skript zur Vorlesung Plastizitaetstheorie SS 2014
        [2] = Cowin, S.C., 1989. Properties of the anisotropic elasticity
        tensor. The Quarterly Journal of Mechanics and Applied Mathematics,
        42(2), pp.249-266.
        [3] Fedorov, F.I., 1968. Theory of elastic waves in crystals.

        Returns
        -------
        np.array with shape (6, 3, 3)
                B(i, :, :) is the i-th dyade of the base.
        '''

        B = np.zeros(
                (self.DIM_MANDEL6, self.DIM, self.DIM),
                dtype=self.dtype,
                )

        B[0, 0, 0] = 1.
        B[1, 1, 1] = 1.
        B[2, 2, 2] = 1.
        B[3, 1, 2] = B[3, 2, 1] = self.factor
        B[4, 0, 2] = B[4, 2, 0] = self.factor
        B[5, 0, 1] = B[5, 1, 0] = self.factor
        return B

    def get_mandel_base_skw(self,):
        '''Calc orthonormal base for possibly non-symmetric second order tensors
        following [4]

        [4] = https://csmbrannon.net/tag/mandel-notation/

        This base can be used to transform
            tensors of second order into vectors with 9 components
            tensors of fourth order into (9 x 9) matrices

        Returns
        -------
        np.array with shape (9, 3, 3)
                B(i, :, :) is the i-th dyade of the base.
        '''

        B = np.zeros(
                (self.DIM_MANDEL9, self.DIM, self.DIM),
                dtype=self.dtype,
                )
        B[0:6, :, :] = self.get_mandel_base_sym()

        B[6, 1, 2] = -self.factor
        B[6, 2, 1] = self.factor
        B[7, 0, 2] = self.factor
        B[7, 2, 0] = -self.factor
        B[8, 0, 1] = -self.factor
        B[8, 1, 0] = self.factor

        return B

    def get_type_by_shape(self, inp):
        '''Identify type depending on inp.shape

        Parameters
        ----------
        inp : np.array with unknown shape
            Representation of tensor/mandel6/mandel9.

        Returns
        -------
        string
            Descriptor of type
        '''

        dim = (self.DIM,)
        dim_mandel6 = (self.DIM_MANDEL6,)
        dim_mandel9 = (self.DIM_MANDEL9,)

        types = {
                2*dim:           't_2',
                4*dim:           't_4',
                1*dim_mandel6:   'm6_2',
                2*dim_mandel6:   'm6_4',
                1*dim_mandel9:   'm9_2',
                2*dim_mandel9:   'm9_4',
                }
        return types[inp.shape]

    def to_mandel6(self, inp):
        '''Identify suitable transformation function and apply it to inp

        Parameters
        ----------
        inp : np.array with unknown shape
            Representation to be transformed into Mandel notation.

        Returns
        -------
        np.array
                Tensor in Mandel notation.
        '''

        f = self.get_to_mandel6_func(inp=inp)
        return f(inp=inp)

    def to_tensor(self, inp):
        '''Identify suitable transformation function and apply it to inp

        Parameters
        ----------
        inp : np.array with unknown shape
            Representation to be transformed into tensor notation.

        Returns
        -------
        np.array
                Tensor.
        '''

        f = self.get_to_tensor_func(inp=inp)
        return f(inp=inp)

    def get_to_mandel6_func(self, inp):
        '''Identify suitable transformation function depending on type

        Parameters
        ----------
        inp : np.array with unknown shape
            Representation to be transformed.

        Returns
        -------
        function handler
                Function suitable to transform inp.
        '''

        type_ = self.get_type_by_shape(inp)

        functions = {
                't_2':      self.tensor2_to_mandel6,
                't_4':      self.tensor4_to_mandel6,
                'm6_2':     self.pass_through,
                'm6_4':     self.pass_through,
                }
        return functions[type_]

    def get_to_tensor_func(self, inp):
        '''Identify suitable transformation function depending on type

        Parameters
        ----------
        inp : np.array with unknown shape
            Representation to be transformed.

        Returns
        -------
        function handler
                Function suitable to transform inp.
        '''

        type_ = self.get_type_by_shape(inp)

        functions = {
                't_2':      self.pass_through,
                't_4':      self.pass_through,
                'm6_2':     self.mandel2_to_tensor,
                'm6_4':     self.mandel4_to_tensor,
                }
        return functions[type_]

    def pass_through(self, inp):
        '''Do nothing, return argument'''
        return inp

    def tensor2_to_mandel6(self, inp):
        '''Transform tensor of second order.

        Parameters
        ----------
        inp : np.array with shape (3, 3)
                Tensor
        Returns
        -------
        np.array with shape (DIM_MANDEL6,)
                Tensor in Mandel notation.
        '''

        out = np.einsum(
                    'aij, ij ->a',
                    self.BASE6,
                    inp,
                    )
        return out

    def tensor4_to_mandel6(self, inp):
        '''Transform tensor of fourth order.

        Parameters
        ----------
        inp : np.array with shape (3, 3, 3, 3)
                Tensor
        Returns
        -------
        np.array with shape (DIM_MANDEL6, DIM_MANDEL6)
                Tensor in Mandel notation.
        '''

        out = np.einsum(
                    'aij, ijkl, bkl ->ab',
                    self.BASE6,
                    inp,
                    self.BASE6,
                    )
        return out

    def mandel2_to_tensor(self, inp):
        '''Transform mandel of first order to tensor of second order.

        Parameters
        ----------
        inp : np.array with shape (DIM_MANDEL6,)
                Mandel representation
        Returns
        -------
        np.array with shape (3, 3)
                Tensor in tensor notation.
        '''

        out = np.einsum(
                    'ajk, a->jk',
                    self.BASE6,
                    inp,
                    )
        return out

    def mandel4_to_tensor(self, inp):
        '''Transform mandel of second order to tensor of fourth order.

        Parameters
        ----------
        inp : np.array with shape (DIM_MANDEL6, DIM_MANDEL6,)
                Mandel representation
        Returns
        -------
        np.array with shape (3, 3, 3, 3)
                Tensor in tensor notation.
        '''

        out = np.einsum(
                    'ajk, ab, bmn->jkmn',
                    self.BASE6,
                    inp,
                    self.BASE6,
                    )
        return out


class VoigtConverter(Converter):
    '''Converter with additional methods handling Voigt notation'''

    def __init__(self, silent=False):

        if not silent:
            print('\nWarning:\n'
                  'Use Voigt-representations only in functions involving\n'
                  '"voigt_" in the function name.\n')

        self.type = 'Voigt'

        self.shear = np.s_[3:6]
        self.quadrant1 = np.s_[0:3, 0:3]
        self.quadrant2 = np.s_[0:3, 3:6]
        self.quadrant3 = np.s_[3:6, 0:3]
        self.quadrant4 = np.s_[3:6, 3:6]

        self.factors_mandel_to_voigt = {
                'stress': [
                        (self.shear,        1./np.sqrt(2.)),
                        ],
                'strain': [
                        (self.shear,        np.sqrt(2.)),
                        ],
                'stiffness': [
                        (self.quadrant2,    1./np.sqrt(2.)),
                        (self.quadrant3,    1./np.sqrt(2.)),
                        (self.quadrant4,    1./2.),
                        ],
                'compliance': [
                        (self.quadrant2,    np.sqrt(2.)),
                        (self.quadrant3,    np.sqrt(2.)),
                        (self.quadrant4,    2.),
                        ],
                }

        super().__init__()

    def mandel_to_voigt(self, mandel, voigt_type):
        '''Transform mandel representation to Voigt depending on voigt_type.

        Parameters
        ----------
        mandel : np.array with shape (6,) or (6, 6) consistent with voigt_type
                Mandel representation

        voigt_type : string
                Defines conversion as types are converted differently.
                Supported types are
                ['stress', 'strain', 'stiffness', 'compliance'].
        Returns
        -------
        np.array with same shape as mandel
                Representation in Voigt notation.
        '''

        voigt = mandel.copy()
        for position, factor in self.factors_mandel_to_voigt[voigt_type]:
            voigt[position] = mandel[position] * factor

        return voigt

    def voigt_to_mandel(self, voigt, voigt_type):
        '''Transform Voigt representation to Mandel depending on voigt_type.

        Parameters
        ----------
        voigt : np.array with shape (6,) or (6, 6) consistent with voigt_type
                Voigt representation

        voigt_type : string
                Defines conversion as types are converted differently.
                Supported types are
                ['stress', 'strain', 'stiffness', 'compliance'].
        Returns
        -------
        np.array with same shape as mandel
                Representation in Mandel notation.
        '''

        mandel = voigt.copy()
        for position, factor in self.factors_mandel_to_voigt[voigt_type]:
            mandel[position] = voigt[position] * 1./factor

        return mandel


if __name__ == '__main__':
    pass
