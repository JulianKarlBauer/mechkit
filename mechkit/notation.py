#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Notation'''

import numpy as np
from mechkit.utils import Ex


class Converter(object):
    r'''
    Convert numerical tensors from one notation to another.

    Supported notations and shapes:

    - tensor

        - 2. order tensor: (3, 3)
        - 4. order tensor: (3, 3, 3, 3,)

    - mandel6 [Mandel1965]_

        - 2. order tensor: (6,)      [Symmetry]
        - 4. order tensor: (6, 6)    [Left- and right- minor symmetry]

    - mandel9 [Brannon2018]_

        - 2. order tensor: (9,)
        - 4. order tensor: (9, 9)

    References and theory can be found in the method descriptions below.

    Methods
    -------
    to_tensor(inp)
        Convert to tensor notation

    to_mandel6(inp)
        Convert to Mandel notation with 6 symmetric base dyads

    to_mandel9(inp)
        Convert to Mandel notation with 6 symmetric and 3 skew base dyads

    to_like(inp, like)
        Convert input to notation of like

    Examples
    --------
    >>> import numpy as np
    >>> import mechkit
    >>> con = mechkit.notation.Converter()
    >>> tensors = mechkit.tensors.Basic()

    >>> t2 = np.array(
    >>>     [[1., 6., 5., ],
    >>>      [6., 2., 4., ],
    >>>      [5., 4., 3., ], ]
    >>>      )
    >>> con.to_mandel6(t2)
    [1.   2.   3.   5.66 7.07 8.49]

    >>> np.sqrt(2.)
    1.414213

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
        r'''Get orthonormal basis of Mandel6 representation introduced by
        [Mandel1965]_, [Fedorov1968]_, [Mehrabadi1990]_  and
        discussed by [Cowin1992]_.

        Base dyads:

        .. math::
            \begin{align*}
                \boldsymbol{B}_1 &= \boldsymbol{e}_1 \otimes \boldsymbol{e}_1\\
                \boldsymbol{B}_2 &= \boldsymbol{e}_2 \otimes \boldsymbol{e}_2\\
                \boldsymbol{B}_3 &= \boldsymbol{e}_3 \otimes \boldsymbol{e}_3\\
                \boldsymbol{B}_4 &= \frac{\sqrt{2}}{2}\left(
                        \boldsymbol{e}_3 \otimes \boldsymbol{e}_2
                        +
                        \boldsymbol{e}_2 \otimes \boldsymbol{e}_3
                        \right)                                              \\
                \boldsymbol{B}_5 &= \frac{\sqrt{2}}{2}\left(
                        \boldsymbol{e}_1 \otimes \boldsymbol{e}_3
                        +
                        \boldsymbol{e}_3 \otimes \boldsymbol{e}_1
                        \right)                                              \\
                \boldsymbol{B}_6 &= \frac{\sqrt{2}}{2}\left(
                        \boldsymbol{e}_2 \otimes \boldsymbol{e}_1
                        +
                        \boldsymbol{e}_1 \otimes \boldsymbol{e}_2
                        \right)
            \end{align*}

        with

            - :math:`\otimes` : Dyadic product
            - :math:`\boldsymbol{e}_\text{i}` : i-th Vector of orthonormal basis

        Orthogonality:

        .. math::
            \begin{align*}
                        \boldsymbol{B}_{\alpha} &\cdot \boldsymbol{B}_{\beta}
                                = \delta_{\alpha\beta}
            \end{align*}

        Conversions: (Einstein notation applies)

        .. math::
            \begin{align*}
                \sigma_{\alpha}^{\text{M}} &=
                    \boldsymbol{\sigma}
                    \cdot
                    \boldsymbol{B}_{\alpha}    \\
                C_{\alpha\beta}^{\text{M}} &=
                    \boldsymbol{B}_{\alpha}
                    \cdot
                    \mathbb{C} \left[\boldsymbol{B}_{\beta}\right]    \\
                \boldsymbol{\sigma} &=
                    \sigma_{\alpha}^{\text{M}}
                    \boldsymbol{B}_{\alpha}    \\
                \mathbb{C} &=
                    C_{\alpha\beta}^{\text{M}}
                    \boldsymbol{B}_{\alpha}
                    \otimes
                    \boldsymbol{B}_{\beta}   \\
            \end{align*}

        with

            - :math:`\boldsymbol{\sigma}` : Second order tensor
            - :math:`\mathbb{C}` : Fourth order tensor
            - :math:`\sigma_{\alpha}^{\text{M}}` : Component in Mandel notation
            - :math:`C_{\alpha\beta}^{\text{M}}` : Component in Mandel notation

        Implications of the Mandel basis:

            - Stress and strain are converted equally, as well as stiffness and compliance. This is in contrast to non-normalized Voigt notation, where conversion rules depend on the physical type of the tensor entity.
            - Eigenvalues and eigenvectors of a component matrix in Mandel notation are equal to eigenvalues and eigenvectors of the tensor.
            - Components of the stress and strain vectors:

        .. math::
            \begin{align*}
                \boldsymbol{\sigma}^{\text{M6}}
                =
                \begin{bmatrix}
                    \sigma_{\text{11}}  \\
                    \sigma_{\text{22}}  \\
                    \sigma_{\text{33}}  \\
                    \frac{\sqrt{2}}{2}\left(
                        \sigma_{\text{32}}
                        +
                        \sigma_{\text{23}}
                    \right)             \\
                    \frac{\sqrt{2}}{2}\left(
                        \sigma_{\text{13}}
                        +
                        \sigma_{\text{31}}
                    \right)             \\
                    \frac{\sqrt{2}}{2}\left(
                        \sigma_{\text{23}}
                        +
                        \sigma_{\text{12}}
                    \right)
                \end{bmatrix}
                &\quad
               \boldsymbol{\varepsilon}^{\text{M6}}
               =
               \begin{bmatrix}
                   \varepsilon_{\text{11}}  \\
                   \varepsilon_{\text{22}}  \\
                   \varepsilon_{\text{33}}  \\
                   \frac{\sqrt{2}}{2}\left(
                       \varepsilon_{\text{32}}
                       +
                       \varepsilon_{\text{23}}
                   \right)             \\
                   \frac{\sqrt{2}}{2}\left(
                       \varepsilon_{\text{13}}
                       +
                       \varepsilon_{\text{31}}
                   \right)             \\
                   \frac{\sqrt{2}}{2}\left(
                       \varepsilon_{\text{23}}
                       +
                       \varepsilon_{\text{12}}
                   \right)
               \end{bmatrix}
            \end{align*}

        .. warning::

            - (Most) unsymmetric parts are discarded during conversion (Exception: Major symmetry of fourth order tensors). Use Mandel9 notation to represent unsymmetric tensors.
            - Components of stiffness matrix in Mandel notation differ from those in Voigt notation. See examples of VoigtConverter below.

        .. rubric:: References

        .. [Mandel1965] Mandel, J., 1965.
            Généralisation de la théorie de plasticité de WT Koiter.
            International Journal of Solids and structures, 1(3), pp.273-295.

        .. [Fedorov1968] Fedorov, F.I., 1968.
            Theory of elastic waves in crystals.

        .. [Mehrabadi1990] Mehrabadi, M.M. and Cowin, S.C., 1990.
            Eigentensors of linear anisotropic elastic materials.
            The Quarterly Journal of Mechanics and Applied Mathematics, 43(1),
            pp.15-41.

        .. [Cowin1992] Cowin, S.C. and Mehrabadi, M.M., 1992.
            The structure of the linear anisotropic elastic symmetries.
            Journal of the Mechanics and Physics of Solids, 40(7),
            pp.1459-1471.

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
        r'''
        Get orthonormal basis of Mandel9 representation [csmbrannonMandel]_,
        [Brannon2018]_. The basis of Mandel6 representation is extended by

        .. math::
            \begin{align*}
                \boldsymbol{B}_7 &= \frac{\sqrt{2}}{2}\left(
                        \boldsymbol{e}_3 \otimes \boldsymbol{e}_2
                        -
                        \boldsymbol{e}_2 \otimes \boldsymbol{e}_3
                        \right)                                              \\
                \boldsymbol{B}_8 &= \frac{\sqrt{2}}{2}\left(
                        \boldsymbol{e}_1 \otimes \boldsymbol{e}_3
                        -
                        \boldsymbol{e}_3 \otimes \boldsymbol{e}_1
                        \right)                                              \\
                \boldsymbol{B}_9 &= \frac{\sqrt{2}}{2}\left(
                        \boldsymbol{e}_2 \otimes \boldsymbol{e}_1
                        -
                        \boldsymbol{e}_1 \otimes \boldsymbol{e}_2
                        \right)
            \end{align*}

        This basis is used to represent skew tensors and implies:

        .. math::
            \begin{align*}
                \boldsymbol{\sigma}^{\text{M9}}
                =
                \begin{bmatrix}
                    \boldsymbol{\sigma}^{\text{M6}}             \\
                    \frac{\sqrt{2}}{2}\left(
                        \sigma_{\text{32}}
                        -
                        \sigma_{\text{23}}
                    \right)             \\
                    \frac{\sqrt{2}}{2}\left(
                        \sigma_{\text{13}}
                        -
                        \sigma_{\text{31}}
                    \right)             \\
                    \frac{\sqrt{2}}{2}\left(
                        \sigma_{\text{23}}
                        -
                        \sigma_{\text{12}}
                    \right)
                \end{bmatrix}
            \end{align*}

        .. rubric:: References

        .. [csmbrannonMandel] https://csmbrannon.net/tag/mandel-notation/

        .. [Brannon2018] Brannon, R.M., 2018. Rotation, Reflection, and Frame
           Changes; Orthogonal tensors in computational engineering mechanics.
           Rotation, Reflection, and Frame Changes; Orthogonal tensors in
           computational engineering mechanics, by Brannon, RM
           ISBN: 978-0-7503-1454-1.
           IOP ebooks. Bristol, UK: IOP Publishing, 2018.


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

    def to_like(self, inp, like):
        '''Convert input to notation of like

        Parameters
        ----------
        inp : np.array with unknown shape
            Input

        like : np.array with unknown shape
            Tensor in desired notation

        Returns
        -------
        np.array
            Input in notation of like
        '''

        type_like = self._get_type_by_shape(like)

        functions = {
                't_':           self.to_tensor,
                'm6':           self.to_mandel6,
                'm9':           self.to_mandel9,
                }

        return functions[type_like[0:2]](inp)

    def _get_type_by_shape(self, inp):
        '''Identify type depending on inp.shape

        Parameters
        ----------
        inp : np.array with unknown shape
            Representation of tensor/mandel6/mandel9

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
    r'''
    Extended converter handling Voigt notation.

    Voigt notation for the following physical quantities is supported:

    - stress
    - strain
    - stiffness
    - compliance

    .. warning::

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

    >>> import mechkit
    >>> con = mechkit.notation.VoigtConverter()

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

        super(type(self), self).__init__()

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

    import mechkit
    con = mechkit.notation.Converter()
    tensors = mechkit.tensors.Basic()

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

    import mechkit
    con = mechkit.notation.VoigtConverter()

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
