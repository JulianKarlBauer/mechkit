#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Notation'''

import numpy as np
from mechkit.utils import Ex

class Converter(object):
    r'''
    Converter to change between common notations of numerical tensors.

    Supported notations

    - tensor

        - 2 order tensor: (3, 3)
        - 4 order tensor: (3, 3, 3, 3,)

    - mandel6

        - 2 order tensor: (6,)      [(symmetric)]
        - 4 order tensor: (6, 6)    [(left- and right- symmetric)]

    - mandel9

        - 2 order tensor: (9,)
        - 4 order tensor: (9, 9)

    Base dyads:

    .. math::
        \begin{align*}
            \boldsymbol{B}_1 &= \boldsymbol{e}_1 \otimes \boldsymbol{e}_1    \\
            \boldsymbol{B}_2 &= \boldsymbol{e}_2 \otimes \boldsymbol{e}_2    \\
            \boldsymbol{B}_3 &= \boldsymbol{e}_3 \otimes \boldsymbol{e}_3    \\
            \boldsymbol{B}_4 &= \frac{\sqrt{2}}{2}\left(
                    \boldsymbol{e}_2 \otimes \boldsymbol{e}_3
                    +
                    \boldsymbol{e}_3 \otimes \boldsymbol{e}_2
                    \right)                                                 \\
            \boldsymbol{B}_5 &= \frac{\sqrt{2}}{2}\left(
                    \boldsymbol{e}_1 \otimes \boldsymbol{e}_3
                    +
                    \boldsymbol{e}_3 \otimes \boldsymbol{e}_1
                    \right)                                                 \\
            \boldsymbol{B}_6 &= \frac{\sqrt{2}}{2}\left(
                    \boldsymbol{e}_1 \otimes \boldsymbol{e}_2
                    +
                    \boldsymbol{e}_2 \otimes \boldsymbol{e}_1
                    \right)                                                 \\
            \boldsymbol{B}_7 &= \frac{\sqrt{2}}{2}\left(
                    -\boldsymbol{e}_2 \otimes \boldsymbol{e}_3
                    +
                    \boldsymbol{e}_3 \otimes \boldsymbol{e}_2
                    \right)                                                 \\
            \boldsymbol{B}_8 &= \frac{\sqrt{2}}{2}\left(
                    \boldsymbol{e}_1 \otimes \boldsymbol{e}_3
                    -
                    \boldsymbol{e}_3 \otimes \boldsymbol{e}_1
                    \right)                                                 \\
            \boldsymbol{B}_9 &= \frac{\sqrt{2}}{2}\left(
                    -\boldsymbol{e}_1 \otimes \boldsymbol{e}_2
                    +
                    \boldsymbol{e}_2 \otimes \boldsymbol{e}_1
                    \right)
        \end{align*}

    Orthogonality:

    .. math::
        \begin{align*}
                    \boldsymbol{B}_{\alpha} &\cdot \boldsymbol{B}_{\beta}
                            = \delta_{\alpha\beta}
        \end{align*}

    Conversions: (Einstein notation applies)

    .. math::
        \begin{align*}
            \sigma_{\alpha} &=
                \boldsymbol{\sigma}
                \cdot
                \boldsymbol{B}_{\alpha}    \\
            C_{\alpha\beta} &=
                \boldsymbol{B}_{\alpha}
                \cdot
                \mathbb{C} \left[\boldsymbol{B}_{\beta}\right]    \\
            \boldsymbol{\sigma} &=
                \sigma_{\alpha}
                \boldsymbol{B}_{\alpha}    \\
            \mathbb{C} &=
                C_{\alpha\beta}
                \boldsymbol{B}_{\alpha}
                \otimes
                \boldsymbol{B}_{\beta}   \\
        \end{align*}

    With:

        - :math:`\sigma_{\alpha}` : Component of second order tensor in Mandel notation
        - :math:`\boldsymbol{\sigma}` : Second order tensor
        - :math:`C_{\alpha\beta}` : Component of fourth order tensor
        - :math:`\mathbb{C}` : Fourth order tensor




    Methods
    -------
    to_tensor(inp)
        Convert to tensor notation

    to_mandel6(inp)
        Convert to Mandel notation with 6 symmetric base dyads

    to_mandel9(inp)
        Convert to Mandel notation with 6 symmetric and 3 skew base dyads

    Examples
    --------
    >>> import mechkit as mk
    >>> con = mk.notation.Converter()
    >>> tensors = mk.tensors.basic()

    >>> tensors.I2
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    >>> con.to_mandel6(tensors.I2)
    [1. 1. 1. 0. 0. 0.]

    >>> np.arange(9).reshape(3,3)
    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    >>> con.to_mandel6(np.arange(9).reshape(3,3))
    [0.   4.   8.   8.49 5.66 2.83]

    >>> tensors.I4s
    [[[[1.  0.  0. ]
       [0.  0.  0. ]
       [0.  0.  0. ]]
      [[0.  0.5 0. ]
       [0.5 0.  0. ]
       [0.  0.  0. ]]
      [[0.  0.  0.5]
       [0.  0.  0. ]
       [0.5 0.  0. ]]]
     [[[0.  0.5 0. ]
       [0.5 0.  0. ]
       [0.  0.  0. ]]
      [[0.  0.  0. ]
       [0.  1.  0. ]
       [0.  0.  0. ]]
      [[0.  0.  0. ]
       [0.  0.  0.5]
       [0.  0.5 0. ]]]
     [[[0.  0.  0.5]
       [0.  0.  0. ]
       [0.5 0.  0. ]]
      [[0.  0.  0. ]
       [0.  0.  0.5]
       [0.  0.5 0. ]]
      [[0.  0.  0. ]
       [0.  0.  0. ]
       [0.  0.  1. ]]]]
    >>> con.to_mandel6(tensors.I4s)
    [[1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 1.]]
    >>> con.to_mandel9(tensors.I4s)
    [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

    >>> #Asymmetric identity
    >>> con.to_mandel9(tensors.I4a)
    [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1.]]

    '''

    def __init__(self, dtype='float64'):

        self.dtype = dtype
        self.factor = np.sqrt(2.) / 2.

        self.DIM = 3
        self.DIM_MANDEL6 = 6
        self.DIM_MANDEL9 = 9
        self.SLICE6 = np.s_[0:6]
        self.BASE6 = self.get_mandel_base_sym()
        self.BASE9 = self.get_mandel_base_skw()

    def get_mandel_base_sym(self,):
        '''Get orthonormal base for symmetric second order tensors following
        [1], [2]

        This base can be used to transform

        - symmetric tensors of second order into vectors with 6 components
        - tensors of fourth order with minor symmetries into (6 x 6) matrices

        .. [1] Böhlke, T., Skript zur Vorlesung Plastizitaetstheorie SS 2014

        .. [2] Fedorov, F.I., 1968. Theory of elastic waves in crystals.

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
        '''Get orthonormal base for possibly non-symmetric second order tensors
        following [4]

        .. [4] https://csmbrannon.net/tag/mandel-notation/

        This base can be used to transform

        - tensors of second order into vectors with 9 components
        - tensors of fourth order into (9 x 9) matrices

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

    def to_mandel6(self, inp, verbose=False):
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

        if verbose:
            print('Skew parts are lost!')

        f = self._get_to_mandel6_func(inp=inp)
        return f(inp=inp)

    def to_mandel9(self, inp):
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

        f = self._get_to_mandel9_func(inp=inp)
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

        f = self._get_to_tensor_func(inp=inp)
        return f(inp=inp)

    def _get_type_by_shape(self, inp):
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

        # assert(inp.shape in types), 'Tensor shape not supported'    \
        #                             '\n Supported shapes: {}'.format(types)

        try:
            type_ = types[inp.shape]
        except KeyError:
            raise Ex('Tensor shape not supported.'
                     '\n Supported shapes: {}'.format(types)
                     )
        return type_

    def _get_to_mandel6_func(self, inp):
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

        type_ = self._get_type_by_shape(inp)

        functions = {
                't_2':      self._tensor2_to_mandel6,
                't_4':      self._tensor4_to_mandel6,
                'm6_2':     self._pass_through,
                'm6_4':     self._pass_through,
                'm9_2':     self._mandel9_2_to_mandel6,
                'm9_4':     self._mandel9_4_to_mandel6,
                }
        return functions[type_]

    def _get_to_mandel9_func(self, inp):
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

        type_ = self._get_type_by_shape(inp)

        functions = {
                't_2':      self._tensor2_to_mandel9,
                't_4':      self._tensor4_to_mandel9,
                'm6_2':     self._mandel6_2_to_mandel9,
                'm6_4':     self._mandel6_4_to_mandel9,
                'm9_2':     self._pass_through,
                'm9_4':     self._pass_through,
                }
        return functions[type_]

    def _get_to_tensor_func(self, inp):
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

        type_ = self._get_type_by_shape(inp)

        functions = {
                't_2':      self._pass_through,
                't_4':      self._pass_through,
                'm6_2':     self._mandel6_2_to_tensor,
                'm6_4':     self._mandel6_4_to_tensor,
                'm9_2':     self._mandel9_2_to_tensor,
                'm9_4':     self._mandel9_4_to_tensor,
                }
        return functions[type_]

    def _pass_through(self, inp):
        '''Do nothing, return argument'''
        return inp

    def _tensor2_to_mandel(self, inp, base):
        out = np.einsum(
                    'aij, ij ->a',
                    base,
                    inp,
                    )
        return out

    def _tensor4_to_mandel(self, inp, base):
        out = np.einsum(
                    'aij, ijkl, bkl ->ab',
                    base,
                    inp,
                    base,
                    )
        return out

    def _tensor2_to_mandel6(self, inp):
        return self._tensor2_to_mandel(inp=inp, base=self.BASE6)

    def _tensor2_to_mandel9(self, inp):
        return self._tensor2_to_mandel(inp=inp, base=self.BASE9)

    def _tensor4_to_mandel6(self, inp):
        return self._tensor4_to_mandel(inp=inp, base=self.BASE6)

    def _tensor4_to_mandel9(self, inp):
        return self._tensor4_to_mandel(inp=inp, base=self.BASE9)

    def _mandel_2_to_tensor(self, inp, base):
        out = np.einsum(
                    'ajk, a->jk',
                    base,
                    inp,
                    )
        return out

    def _mandel_4_to_tensor(self, inp, base):
        out = np.einsum(
                    'ajk, ab, bmn->jkmn',
                    base,
                    inp,
                    base,
                    )
        return out

    def _mandel6_2_to_tensor(self, inp):
        return self._mandel_2_to_tensor(inp=inp, base=self.BASE6)

    def _mandel6_4_to_tensor(self, inp):
        return self._mandel_4_to_tensor(inp=inp, base=self.BASE6)

    def _mandel9_2_to_tensor(self, inp):
        return self._mandel_2_to_tensor(inp=inp, base=self.BASE9)

    def _mandel9_4_to_tensor(self, inp):
        return self._mandel_4_to_tensor(inp=inp, base=self.BASE9)

    def _mandel6_2_to_mandel9(self, inp):
        zeros = np.zeros((self.DIM_MANDEL9, ), dtype=self.dtype)
        zeros[self.SLICE6] = inp
        return zeros

    def _mandel6_4_to_mandel9(self, inp):
        zeros = np.zeros(
                    (self.DIM_MANDEL9, self.DIM_MANDEL9),
                    dtype=self.dtype,
                    )
        zeros[self.SLICE6, self.SLICE6] = inp
        return zeros

    def _mandel9_2_to_mandel6(self, inp):
        return inp[self.SLICE6]

    def _mandel9_4_to_mandel6(self, inp):
        return inp[self.SLICE6, self.SLICE6]


class VoigtConverter(Converter):
    '''
    Extended converter handling Voigt notation

    Voigt notation for the following physical quantities are supported:

    - stress
    - strain
    - stiffness
    - compliance

    Warning
    =======

    Usage of Voigt-representations is highly discouraged.
    Don't use representations in Voigt notation in function
    lacking "voigt" in the method name.
    The results will be wrong.

    Tensor representations in Voigt notation have the same
    dimensions than those in Mandel6 notation and therefore are
    treated as representations in Mandel6 notation, when passed
    to methods not including "voigt" in the method name.

    Methods
    -------
    mandel6_to_voigt(inp, voigt_type)
        Convert from Mandel6 to Voigt notation based on physical meaning of inp
    voigt_to_mandel6(inp, voigt_type)
        Convert from Voigt to Mandel6 notation based on physical meaning of inp

    Examples
    --------

    >>> import mechkit as mk
    >>> con = mk.notation.VoigtConverter()

    >>> ones_2 = np.ones((3, 3),)
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    >>> ones_2_mandel = con.to_mandel6(ones_2)
    [1.   1.   1.   1.41 1.41 1.41]
    >>> con.mandel6_to_voigt(inp=ones_2_mandel, voigt_type='stress')
    [1. 1. 1. 1. 1. 1.]
    >>> con.mandel6_to_voigt(inp=ones_2_mandel, voigt_type='strain')
    [1. 1. 1. 2. 2. 2.]

    >>> ones_4_mandel = con.to_mandel6(np.ones((3, 3, 3, 3),))
    [[1.   1.   1.   1.41 1.41 1.41]
     [1.   1.   1.   1.41 1.41 1.41]
     [1.   1.   1.   1.41 1.41 1.41]
     [1.41 1.41 1.41 2.   2.   2.  ]
     [1.41 1.41 1.41 2.   2.   2.  ]
     [1.41 1.41 1.41 2.   2.   2.  ]]
    >>> con.mandel6_to_voigt(inp=ones_4_mandel, voigt_type='stiffness')
    [[1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]]
    >>> con.mandel6_to_voigt(inp=ones_4_mandel, voigt_type='compliance')
    [[1. 1. 1. 2. 2. 2.]
     [1. 1. 1. 2. 2. 2.]
     [1. 1. 1. 2. 2. 2.]
     [2. 2. 2. 4. 4. 4.]
     [2. 2. 2. 4. 4. 4.]
     [2. 2. 2. 4. 4. 4.]]

    '''
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
    # Examples

    np.set_printoptions(
            linewidth=140,
            precision=2,
            # suppress=False,
            )

    # Converter

    import mechkit as mk
    con = mk.notation.Converter()
    tensors = mk.tensors.basic()

    printQueue = [
            # import mechkit as mk
            'tensors.I2',
            'con.to_mandel6(tensors.I2)',
            'np.arange(9).reshape(3,3)',
            'con.to_mandel6(np.arange(9).reshape(3,3))',
            'tensors.I4s',
            'con.to_mandel6(tensors.I4s)',
            'con.to_mandel9(tensors.I4s)',
            'con.to_mandel9(tensors.I4s)',
            ]
    for val in printQueue:
        print(val)
        print(eval(val), '\n')

    # VoigtConverter

    import mechkit as mk
    con = mk.notation.VoigtConverter()

    ones_2 = np.ones((3, 3),)
    ones_2_mandel = con.to_mandel6(ones_2)
    ones_4_mandel = con.to_mandel6(np.ones((3, 3, 3, 3),))

    printQueue = [
            'ones_2',
            'ones_2_mandel',
            "con.mandel6_to_voigt(inp=ones_2_mandel, voigt_type='stress')",
            "con.mandel6_to_voigt(inp=ones_2_mandel, voigt_type='strain')",
            'ones_4_mandel',
            "con.mandel6_to_voigt(inp=ones_4_mandel, voigt_type='stiffness')",
            "con.mandel6_to_voigt(inp=ones_4_mandel, voigt_type='compliance')",
            ]
    for val in printQueue:
        print(val)
        print(eval(val), '\n')
