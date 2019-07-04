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
        self.SLICE6 = np.s_[0:6]
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
        '''Convert to Mandel6 notation

        Parameters
        ----------
        inp : np.array with unknown shape
            Input

        Returns
        -------
        np.array
            Input in Mandel6 notation
        '''

        f = self.get_to_mandel6_func(inp=inp)
        return f(inp=inp)

    def to_mandel9(self, inp, verbose=False):
        '''Convert to Mandel9 notation

        Parameters
        ----------
        inp : np.array with unknown shape
            Input

        Returns
        -------
        np.array
            Input in Mandel9 notation
        '''

        if verbose:
            print('Skew parts are lost!')

        f = self.get_to_mandel9_func(inp=inp)
        return f(inp=inp)

    def to_tensor(self, inp):
        '''Convert to tensor notation

        Parameters
        ----------
        inp : np.array with unknown shape
            Input

        Returns
        -------
        np.array
            Input in tensor notation
        '''

        f = self.get_to_tensor_func(inp=inp)
        return f(inp=inp)

    def get_to_mandel6_func(self, inp):
        '''Select transformation function by type

        Parameters
        ----------
        inp : np.array with unknown shape
            Input

        Returns
        -------
        function handler
            Function transforming input to Mandel6
        '''

        type_ = self.get_type_by_shape(inp)

        functions = {
                't_2':      self.tensor2_to_mandel6,
                't_4':      self.tensor4_to_mandel6,
                'm6_2':     self.pass_through,
                'm6_4':     self.pass_through,
                'm9_2':     self.mandel9_2_to_mandel6,
                'm9_4':     self.mandel9_4_to_mandel6,
                }
        return functions[type_]

    def get_to_mandel9_func(self, inp):
        '''Select transformation function by type

        Parameters
        ----------
        inp : np.array with unknown shape
            Input

        Returns
        -------
        function handler
            Function transforming input to Mandel9
        '''

        type_ = self.get_type_by_shape(inp)

        functions = {
                't_2':      self.tensor2_to_mandel9,
                't_4':      self.tensor4_to_mandel9,
                'm6_2':     self.mandel6_2_to_mandel9,
                'm6_4':     self.mandel6_4_to_mandel9,
                'm9_2':     self.pass_through,
                'm9_4':     self.pass_through,
                }
        return functions[type_]

    def get_to_tensor_func(self, inp):
        '''Select transformation function by type

        Parameters
        ----------
        inp : np.array with unknown shape
            Input

        Returns
        -------
        function handler
            Function transforming input to tensor
        '''

        type_ = self.get_type_by_shape(inp)

        functions = {
                't_2':      self.pass_through,
                't_4':      self.pass_through,
                'm6_2':     self.mandel6_2_to_tensor,
                'm6_4':     self.mandel6_4_to_tensor,
                'm9_2':     self.mandel9_2_to_tensor,
                'm9_4':     self.mandel9_4_to_tensor,
                }
        return functions[type_]

    def pass_through(self, inp):
        '''Do nothing, return argument'''
        return inp

    def tensor2_to_mandel(self, inp, base):
        out = np.einsum(
                    'aij, ij ->a',
                    base,
                    inp,
                    )
        return out

    def tensor4_to_mandel(self, inp, base):
        out = np.einsum(
                    'aij, ijkl, bkl ->ab',
                    base,
                    inp,
                    base,
                    )
        return out

    def tensor2_to_mandel6(self, inp):
        return self.tensor2_to_mandel(inp=inp, base=self.BASE6)

    def tensor2_to_mandel9(self, inp):
        return self.tensor2_to_mandel(inp=inp, base=self.BASE9)

    def tensor4_to_mandel6(self, inp):
        return self.tensor4_to_mandel(inp=inp, base=self.BASE6)

    def tensor4_to_mandel9(self, inp):
        return self.tensor4_to_mandel(inp=inp, base=self.BASE9)

    def mandel_2_to_tensor(self, inp, base):
        out = np.einsum(
                    'ajk, a->jk',
                    base,
                    inp,
                    )
        return out

    def mandel_4_to_tensor(self, inp, base):
        out = np.einsum(
                    'ajk, ab, bmn->jkmn',
                    base,
                    inp,
                    base,
                    )
        return out

    def mandel6_2_to_tensor(self, inp):
        return self.mandel_2_to_tensor(inp=inp, base=self.BASE6)

    def mandel6_4_to_tensor(self, inp):
        return self.mandel_4_to_tensor(inp=inp, base=self.BASE6)

    def mandel9_2_to_tensor(self, inp):
        return self.mandel_2_to_tensor(inp=inp, base=self.BASE9)

    def mandel9_4_to_tensor(self, inp):
        return self.mandel_4_to_tensor(inp=inp, base=self.BASE9)

    def mandel6_2_to_mandel9(self, inp):
        zeros = np.zeros((self.DIM_MANDEL9, ), dtype=self.dtype)
        zeros[self.SLICE6] = inp
        return zeros

    def mandel6_4_to_mandel9(self, inp):
        zeros = np.zeros(
                    (self.DIM_MANDEL9, self.DIM_MANDEL9),
                    dtype=self.dtype,
                    )
        zeros[self.SLICE6, self.SLICE6] = inp
        return zeros

    def mandel9_2_to_mandel6(self, inp):
        return inp[self.SLICE6]

    def mandel9_4_to_mandel6(self, inp):
        return inp[self.SLICE6, self.SLICE6]


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

    def mandel6_to_voigt(self, inp, voigt_type):
        '''Transform Mandel to Voigt depending on voigt_type.

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
        np.array with same shape as inp
                Voigt representation
        '''

        voigt = inp.copy()
        for position, factor in self.factors_mandel_to_voigt[voigt_type]:
            voigt[position] = inp[position] * factor

        return voigt

    def voigt_to_mandel6(self, inp, voigt_type):
        '''Transform Voigt to Mandel depending on voigt_type.

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
        np.array with same shape as inp
                Mandel representation
        '''

        mandel = inp.copy()
        for position, factor in self.factors_mandel_to_voigt[voigt_type]:
            mandel[position] = inp[position] * 1./factor

        return mandel


if __name__ == '__main__':
    pass
