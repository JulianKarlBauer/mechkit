#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Notations and converters, converting from one notation to another
"""

import numpy as np
from mechkit.utils import Ex
import networkx as nx
import functools
import sys

if sys.version_info > (3, 6):
    import sympy as sp


def get_default_factor():
    return np.sqrt(2.0) / 2.0


def get_mandel_base_sym(dtype="float64", one=1.0, factor=None):
    r"""Get orthonormal basis of Mandel6 representation introduced by
    :cite:p:`Mandel1965`, :cite:p:`Fedorov1968`, :cite:p:`Mehrabadi1990` and
    discussed by :cite:p:`Cowin1992`.

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
                    \sigma_{\text{21}}
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
                   \varepsilon_{\text{21}}
                   +
                   \varepsilon_{\text{12}}
               \right)
           \end{bmatrix}
        \end{align*}

    .. warning::

        - (Most) unsymmetric parts are discarded during conversion (Exception: Major symmetry of fourth order tensors). Use Mandel9 notation to represent unsymmetric tensors.
        - Components of stiffness matrix in Mandel notation differ from those in Voigt notation. See examples of VoigtConverter below.


    Returns
    -------
    np.array with shape (6, 3, 3)
            B(i, :, :) is the i-th dyade of the base.
    """

    if factor is None:
        factor = get_default_factor()

    B = np.zeros((6, 3, 3), dtype=dtype)

    B[0, 0, 0] = one
    B[1, 1, 1] = one
    B[2, 2, 2] = one
    B[3, 1, 2] = B[3, 2, 1] = factor
    B[4, 0, 2] = B[4, 2, 0] = factor
    B[5, 0, 1] = B[5, 1, 0] = factor
    return B


def get_mandel_base_skw(dtype="float64", one=1.0, factor=None):
    r"""
    Get orthonormal basis of Mandel9 representation [csmbrannonMandel]_,
    :cite:p:`Brannon2018`. The basis of Mandel6 representation is extended by

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


    Returns
    -------
    np.array with shape (9, 3, 3)
            B(i, :, :) is the i-th dyade of the base.
    """

    if factor is None:
        factor = get_default_factor()

    B = np.zeros((9, 3, 3), dtype=dtype)
    B[0:6, :, :] = get_mandel_base_sym(dtype=dtype, one=one, factor=factor)

    B[6, 1, 2] = -factor
    B[6, 2, 1] = factor
    B[7, 0, 2] = factor
    B[7, 2, 0] = -factor
    B[8, 0, 1] = -factor
    B[8, 1, 0] = factor
    return B


class Converter(object):
    r"""
    Convert numerical tensors from one notation to another.

    Supported notations and shapes:

    - tensor

        - 2. order tensor: (3, 3)
        - 4. order tensor: (3, 3, 3, 3,)

    - mandel6 :cite:p:`Mandel1965`

        - 2. order tensor: (6,)      [Symmetry]
        - 4. order tensor: (6, 6)    [Left- and right- minor symmetry]

    - mandel9 :cite:p:`Brannon2018`

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

    """

    def __init__(self, dtype="float64", one=1.0, factor=None):

        self.dtype = dtype

        self.DIM = 3
        self.DIM_MANDEL6 = 6
        self.DIM_MANDEL9 = 9
        self.SLICE6 = np.s_[0:6]
        self.BASE6 = get_mandel_base_sym(dtype=dtype, one=one, factor=factor)
        self.BASE9 = get_mandel_base_skw(dtype=dtype, one=one, factor=factor)

    def to_mandel6(self, inp, verbose=False):

        if verbose:
            print("Skew parts are lost!")

        f = self._get_to_mandel6_func(inp=inp)
        return f(inp=inp)

    def to_mandel9(self, inp):

        f = self._get_to_mandel9_func(inp=inp)
        return f(inp=inp)

    def to_tensor(self, inp):

        f = self._get_to_tensor_func(inp=inp)
        return f(inp=inp)

    def to_like(self, inp, like):

        type_like = self._get_type_by_shape(like)

        functions = {
            "t_": self.to_tensor,
            "m6": self.to_mandel6,
            "m9": self.to_mandel9,
        }

        return functions[type_like[0:2]](inp)

    def _get_type_by_shape(self, inp):
        dim = (self.DIM,)
        dim_mandel6 = (self.DIM_MANDEL6,)
        dim_mandel9 = (self.DIM_MANDEL9,)

        types = {
            2 * dim: "t_2",
            4 * dim: "t_4",
            1 * dim_mandel6: "m6_2",
            2 * dim_mandel6: "m6_4",
            1 * dim_mandel9: "m9_2",
            2 * dim_mandel9: "m9_4",
        }

        try:
            type_ = types[inp.shape]
        except KeyError:
            raise Ex(
                "Tensor shape not supported." "\n Supported shapes: {}".format(types)
            )
        return type_

    def _get_to_mandel6_func(self, inp):
        type_ = self._get_type_by_shape(inp)

        functions = {
            "t_2": self._tensor2_to_mandel6,
            "t_4": self._tensor4_to_mandel6,
            "m6_2": self._pass_through,
            "m6_4": self._pass_through,
            "m9_2": self._mandel9_2_to_mandel6,
            "m9_4": self._mandel9_4_to_mandel6,
        }
        return functions[type_]

    def _get_to_mandel9_func(self, inp):
        type_ = self._get_type_by_shape(inp)

        functions = {
            "t_2": self._tensor2_to_mandel9,
            "t_4": self._tensor4_to_mandel9,
            "m6_2": self._mandel6_2_to_mandel9,
            "m6_4": self._mandel6_4_to_mandel9,
            "m9_2": self._pass_through,
            "m9_4": self._pass_through,
        }
        return functions[type_]

    def _get_to_tensor_func(self, inp):
        type_ = self._get_type_by_shape(inp)

        functions = {
            "t_2": self._pass_through,
            "t_4": self._pass_through,
            "m6_2": self._mandel6_2_to_tensor,
            "m6_4": self._mandel6_4_to_tensor,
            "m9_2": self._mandel9_2_to_tensor,
            "m9_4": self._mandel9_4_to_tensor,
        }
        return functions[type_]

    def _pass_through(self, inp):
        """Do nothing, return argument"""
        return inp

    def _tensor2_to_mandel(self, inp, base):
        # out = np.einsum("aij, ij ->a", base, inp)
        out = np.tensordot(base, inp, axes=2)
        return out

    def _tensor4_to_mandel(self, inp, base):
        # out = np.einsum("aij, ijkl, bkl ->ab", base, inp, base)
        tmp = np.tensordot(base, inp, axes=2)
        out = np.tensordot(tmp, np.einsum("bkl->klb", base), axes=2)
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
        # out = np.einsum("ajk, a->jk", base, inp)
        out = np.tensordot(inp, base, axes=1)
        return out

    def _mandel_4_to_tensor(self, inp, base):
        # out = np.einsum("ajk, ab, bmn->jkmn", base, inp, base)
        tmp = np.tensordot(np.einsum("ajk->jka", base), inp, axes=1)
        out = np.tensordot(tmp, base, axes=1)
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
        zeros = np.zeros((self.DIM_MANDEL9,), dtype=self.dtype)
        zeros[self.SLICE6] = inp
        return zeros

    def _mandel6_4_to_mandel9(self, inp):
        zeros = np.zeros((self.DIM_MANDEL9, self.DIM_MANDEL9), dtype=self.dtype)
        zeros[self.SLICE6, self.SLICE6] = inp
        return zeros

    def _mandel9_2_to_mandel6(self, inp):
        return inp[self.SLICE6]

    def _mandel9_4_to_mandel6(self, inp):
        return inp[self.SLICE6, self.SLICE6]


if sys.version_info > (3, 6):

    class ConverterSymbolic(Converter):
        def __init__(self):
            super(type(self), self).__init__(
                dtype=sp.Symbol, one=sp.S(1), factor=sp.sqrt(2) / sp.S(2)
            )


class VoigtConverter(Converter):
    r"""
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

    Component order is defined as

    .. math::
        \begin{align*}
            \boldsymbol{\sigma}^{\text{Voigt}}
            =
            \begin{bmatrix}
                \sigma_{\text{11}}  \\
                \sigma_{\text{22}}  \\
                \sigma_{\text{33}}  \\
                \sigma_{\text{23}}  \\
                \sigma_{\text{13}}  \\
                \sigma_{\text{12}}  \\
            \end{bmatrix}
            &\quad
           \boldsymbol{\varepsilon}^{\text{Voigt}}
           =
           \begin{bmatrix}
                \varepsilon_{\text{11}}  \\
                \varepsilon_{\text{22}}  \\
                \varepsilon_{\text{33}}  \\
                2\varepsilon_{\text{23}}  \\
                2\varepsilon_{\text{13}}  \\
                2\varepsilon_{\text{12}}  \\
           \end{bmatrix}.
        \end{align*}

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

    """

    def __init__(self, silent=False):

        if not silent:
            print(
                "\nWarning:\n"
                "Use Voigt-representations only in functions involving\n"
                '"voigt_" in the function name.\n'
            )

        self.type = "Voigt"

        self.shear = np.s_[3:6]
        self.quadrant1 = np.s_[0:3, 0:3]
        self.quadrant2 = np.s_[0:3, 3:6]
        self.quadrant3 = np.s_[3:6, 0:3]
        self.quadrant4 = np.s_[3:6, 3:6]

        self.factors_mandel_to_voigt = {
            "stress": [(self.shear, 1.0 / np.sqrt(2.0))],
            "strain": [(self.shear, np.sqrt(2.0))],
            "stiffness": [
                (self.quadrant2, 1.0 / np.sqrt(2.0)),
                (self.quadrant3, 1.0 / np.sqrt(2.0)),
                (self.quadrant4, 1.0 / 2.0),
            ],
            "compliance": [
                (self.quadrant2, np.sqrt(2.0)),
                (self.quadrant3, np.sqrt(2.0)),
                (self.quadrant4, 2.0),
            ],
        }

        super(VoigtConverter, self).__init__()

    def mandel6_to_voigt(self, inp, voigt_type):

        voigt = inp.copy()
        for position, factor in self.factors_mandel_to_voigt[voigt_type]:
            voigt[position] = inp[position] * factor

        return voigt

    def voigt_to_mandel6(self, inp, voigt_type):

        mandel = inp.copy()
        for position, factor in self.factors_mandel_to_voigt[voigt_type]:
            mandel[position] = inp[position] * 1.0 / factor

        return mandel


class ExplicitConverter(object):
    r"""
    Vectorized extendable converter.

    As the number of necessary convertion rules between a rising number of notations
    increases fast, a graph/network algorithm is used to identify the shortes
    convertion path between source and target notation.
    Notations represent nodes of a graph and convertion rules represent directed
    edges between nodes.

    Methods
    -------
    convert(inp, target, source, quantity)
        The tensorial input argument `inp`
        is converted from explicitly stated
        `source` notation to the `target` notation.
        As some notations depend on the physical meaning of the tensorial quantity,
        the `quantity` type has to be specified explicitly.
        Supported quantity types are: Stress, strain, stiffness, compliance


    Currently supported notations for all quantity types are:

        - tensor
        - mandel9
        - mandel6
        - voigt
        - umat
        - vumat

    and in addition for quantity type "stiffness":

        - abaqusMaterialAnisotropic

    **UMAT notation**

    .. math::
        \begin{align*}
            \boldsymbol{\sigma}^{\text{UMAT}}
            =
            \begin{bmatrix}
                \sigma_{\text{11}}  \\
                \sigma_{\text{22}}  \\
                \sigma_{\text{33}}  \\
                \sigma_{\text{12}}  \\
                \sigma_{\text{13}}  \\
                \sigma_{\text{23}}  \\
            \end{bmatrix}
            &\quad
           \boldsymbol{\varepsilon}^{\text{UMAT}}
           =
           \begin{bmatrix}
                \varepsilon_{\text{11}}  \\
                \varepsilon_{\text{22}}  \\
                \varepsilon_{\text{33}}  \\
                2\varepsilon_{\text{12}}  \\
                2\varepsilon_{\text{13}}  \\
                2\varepsilon_{\text{23}}  \\
           \end{bmatrix}.
        \end{align*}

    todo: add stiffness and compliance

    **VUMAT notation**

    .. math::
        \begin{align*}
            \boldsymbol{\sigma}^{\text{VUMAT}}
            =
            \begin{bmatrix}
                \sigma_{\text{11}}  \\
                \sigma_{\text{22}}  \\
                \sigma_{\text{33}}  \\
                \sigma_{\text{12}}  \\
                \sigma_{\text{23}}  \\
                \sigma_{\text{13}}  \\
            \end{bmatrix}
            &\quad
           \boldsymbol{\varepsilon}^{\text{VUMAT}}
           =
           \begin{bmatrix}
                \varepsilon_{\text{11}}  \\
                \varepsilon_{\text{22}}  \\
                \varepsilon_{\text{33}}  \\
                \varepsilon_{\text{12}}  \\
                \varepsilon_{\text{23}}  \\
                \varepsilon_{\text{13}}  \\
           \end{bmatrix}.
        \end{align*}

    todo: add stiffness and compliance

    todo: add stiffness abaqusMaterialAnisotropic


    """

    def __init__(self, dtype="float64"):

        self.dtype = dtype
        self.factor = np.sqrt(2.0) / 2.0

        self.DIM = 3
        self.DIM_MANDEL6 = 6
        self.DIM_MANDEL9 = 9

        self.SLICE6 = np.s_[..., 0:6]
        self.SLICE6BY6 = np.s_[..., 0:6, 0:6]
        self.BASE6 = get_mandel_base_sym()
        self.BASE9 = get_mandel_base_skw()

        self.shear = np.s_[..., 3:6]
        self.quadrant1 = np.s_[..., 0:3, 0:3]
        self.quadrant2 = np.s_[..., 0:3, 3:6]
        self.quadrant3 = np.s_[..., 3:6, 0:3]
        self.quadrant4 = np.s_[..., 3:6, 3:6]

        self.factors_mandel_to_voigt = {
            "stress": [(self.shear, 1.0 / np.sqrt(2.0))],
            "strain": [(self.shear, np.sqrt(2.0))],
            "stiffness": [
                (self.quadrant2, 1.0 / np.sqrt(2.0)),
                (self.quadrant3, 1.0 / np.sqrt(2.0)),
                (self.quadrant4, 1.0 / 2.0),
            ],
            "compliance": [
                (self.quadrant2, np.sqrt(2.0)),
                (self.quadrant3, np.sqrt(2.0)),
                (self.quadrant4, 2.0),
            ],
        }

        self.factors_voigt_to_reordered_vumat = {
            "stress": [],
            "strain": [(self.shear, 1.0 / 2.0)],
            "stiffness": [(self.quadrant2, 2.0), (self.quadrant4, 2.0)],
            "compliance": [(self.quadrant3, 1.0 / 2.0), (self.quadrant4, 1.0 / 2.0)],
        }

        self.factors_reordered_vumat_to_voigt = {
            key: [(slic, 1.0 / fac) for slic, fac in val]
            for key, val in self.factors_voigt_to_reordered_vumat.items()
        }

        self.map_voigt_to_abaqusMaterialElasticAnisotropic = [
            (0, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (0, 5),
            (1, 5),
            (2, 5),
            (5, 5),
            (0, 4),
            (1, 4),
            (2, 4),
            (5, 4),
            (4, 4),
            (0, 3),
            (1, 3),
            (2, 3),
            (5, 3),
            (4, 3),
            (3, 3),
        ]

        self.edges_dict = {
            "stress": [
                ("tensor", "mandel6", dict(func=self._tensor_to_mandel6_2)),
                ("tensor", "mandel9", dict(func=self._tensor_to_mandel9_2)),
                ("mandel9", "tensor", dict(func=self._mandel9_to_tensor_2)),
                ("mandel9", "mandel6", dict(func=self._mandel9_to_mandel6_2)),
                ("mandel6", "tensor", dict(func=self._mandel6_to_tensor_2)),
                ("mandel6", "mandel9", dict(func=self._mandel6_to_mandel9_2)),
                ("mandel6", "voigt", dict(func=self._mandel6_to_voigt_stress)),
                ("voigt", "mandel6", dict(func=self._voigt_to_mandel6_stress)),
                ("voigt", "umat", dict(func=self._voigt_umat_2)),
                ("voigt", "vumat", dict(func=self._voigt_to_vumat_stress)),
                ("umat", "voigt", dict(func=self._voigt_umat_2)),
                ("vumat", "voigt", dict(func=self._vumat_to_voigt_stress)),
            ],
            "strain": [
                ("tensor", "mandel6", dict(func=self._tensor_to_mandel6_2)),
                ("tensor", "mandel9", dict(func=self._tensor_to_mandel9_2)),
                ("mandel9", "tensor", dict(func=self._mandel9_to_tensor_2)),
                ("mandel9", "mandel6", dict(func=self._mandel9_to_mandel6_2)),
                ("mandel6", "tensor", dict(func=self._mandel6_to_tensor_2)),
                ("mandel6", "mandel9", dict(func=self._mandel6_to_mandel9_2)),
                ("mandel6", "voigt", dict(func=self._mandel6_to_voigt_strain)),
                ("voigt", "mandel6", dict(func=self._voigt_to_mandel6_strain)),
                ("voigt", "umat", dict(func=self._voigt_umat_2)),
                ("voigt", "vumat", dict(func=self._voigt_to_vumat_strain)),
                ("umat", "voigt", dict(func=self._voigt_umat_2)),
                ("vumat", "voigt", dict(func=self._vumat_to_voigt_strain)),
            ],
            "stiffness": [
                ("tensor", "mandel6", dict(func=self._tensor_to_mandel6_4)),
                ("tensor", "mandel9", dict(func=self._tensor_to_mandel9_4)),
                ("mandel9", "tensor", dict(func=self._mandel9_to_tensor_4)),
                ("mandel9", "mandel6", dict(func=self._mandel9_to_mandel6_4)),
                ("mandel6", "tensor", dict(func=self._mandel6_to_tensor_4)),
                ("mandel6", "mandel9", dict(func=self._mandel6_to_mandel9_4)),
                ("mandel6", "voigt", dict(func=self._mandel6_to_voigt_stiffness)),
                ("voigt", "mandel6", dict(func=self._voigt_to_mandel6_stiffness)),
                ("voigt", "umat", dict(func=self._voigt_umat_4)),
                ("voigt", "vumat", dict(func=self._voigt_to_vumat_stiffness)),
                ("umat", "voigt", dict(func=self._voigt_umat_4)),
                ("vumat", "voigt", dict(func=self._vumat_to_voigt_stiffness)),
                (
                    "voigt",
                    "abaqusMaterialAnisotropic",
                    dict(func=self._voigt_to_abaqusMaterialElasticAnisotropic),
                ),
                (
                    "abaqusMaterialAnisotropic",
                    "voigt",
                    dict(func=self._abaqusMaterialElasticAnisotropic_to_voigt),
                ),
            ],
            "compliance": [
                ("tensor", "mandel6", dict(func=self._tensor_to_mandel6_4)),
                ("tensor", "mandel9", dict(func=self._tensor_to_mandel9_4)),
                ("mandel9", "tensor", dict(func=self._mandel9_to_tensor_4)),
                ("mandel9", "mandel6", dict(func=self._mandel9_to_mandel6_4)),
                ("mandel6", "tensor", dict(func=self._mandel6_to_tensor_4)),
                ("mandel6", "mandel9", dict(func=self._mandel6_to_mandel9_4)),
                ("mandel6", "voigt", dict(func=self._mandel6_to_voigt_compliance)),
                ("voigt", "mandel6", dict(func=self._voigt_to_mandel6_compliance)),
                ("voigt", "umat", dict(func=self._voigt_umat_4)),
                ("voigt", "vumat", dict(func=self._voigt_to_vumat_compliance)),
                ("umat", "voigt", dict(func=self._voigt_umat_4)),
                ("vumat", "voigt", dict(func=self._vumat_to_voigt_compliance)),
            ],
        }

        self.graphs_dict = {
            key: nx.DiGraph(edges) for key, edges in self.edges_dict.items()
        }

    def convert(self, inp, target, source, quantity):

        graph = self.graphs_dict[quantity]

        path = nx.shortest_path(G=graph, source=source, target=target)

        steps = list(nx.utils.pairwise(path))

        new = inp.copy()
        for step_start, step_end in steps:
            func = graph.edges[step_start, step_end]["func"]
            new = func(new)

        return new

    def _tensor_to_mandel_2(self, inp, base):
        out = np.einsum("aij, ...ij ->...a", base, inp)
        return out

    def _tensor_to_mandel_4(self, inp, base):
        out = np.einsum("aij, ...ijkl, bkl ->...ab", base, inp, base)
        return out

    def _tensor_to_mandel6_2(self, inp):
        return self._tensor_to_mandel_2(inp=inp, base=self.BASE6)

    def _tensor_to_mandel9_2(self, inp):
        return self._tensor_to_mandel_2(inp=inp, base=self.BASE9)

    def _tensor_to_mandel6_4(self, inp):
        return self._tensor_to_mandel_4(inp=inp, base=self.BASE6)

    def _tensor_to_mandel9_4(self, inp):
        return self._tensor_to_mandel_4(inp=inp, base=self.BASE9)

    def _mandel_to_tensor_2(self, inp, base):
        out = np.einsum("ajk, ...a->...jk", base, inp)
        return out

    def _mandel_to_tensor_4(self, inp, base):
        out = np.einsum("ajk, ...ab, bmn->...jkmn", base, inp, base)
        return out

    def _mandel6_to_tensor_2(self, inp):
        return self._mandel_to_tensor_2(inp=inp, base=self.BASE6)

    def _mandel6_to_tensor_4(self, inp):
        return self._mandel_to_tensor_4(inp=inp, base=self.BASE6)

    def _mandel9_to_tensor_2(self, inp):
        return self._mandel_to_tensor_2(inp=inp, base=self.BASE9)

    def _mandel9_to_tensor_4(self, inp):
        return self._mandel_to_tensor_4(inp=inp, base=self.BASE9)

    def _mandel6_to_mandel9_2(self, inp):
        shape = inp.shape[:-1] + (self.DIM_MANDEL9,)
        zeros = np.zeros(shape, dtype=self.dtype)
        zeros[self.SLICE6] = inp
        return zeros

    def _mandel6_to_mandel9_4(self, inp):
        shape = inp.shape[:-2] + (self.DIM_MANDEL9, self.DIM_MANDEL9)
        zeros = np.zeros(shape, dtype=self.dtype)
        zeros[self.SLICE6BY6] = inp
        return zeros

    def _mandel9_to_mandel6_2(self, inp):
        return inp[self.SLICE6]

    def _mandel9_to_mandel6_4(self, inp):
        return inp[self.SLICE6BY6]

    def _mandel6_to_voigt(self, inp, quantity):
        voigt = inp.copy()
        for position, factor in self.factors_mandel_to_voigt[quantity]:
            voigt[position] = inp[position] * factor
        return voigt

    def _mandel6_to_voigt_stress(self, inp):
        return self._mandel6_to_voigt(inp=inp, quantity="stress")

    def _mandel6_to_voigt_strain(self, inp):
        return self._mandel6_to_voigt(inp=inp, quantity="strain")

    def _mandel6_to_voigt_stiffness(self, inp):
        return self._mandel6_to_voigt(inp=inp, quantity="stiffness")

    def _mandel6_to_voigt_compliance(self, inp):
        return self._mandel6_to_voigt(inp=inp, quantity="compliance")

    def _voigt_to_mandel6(self, inp, quantity):
        mandel = inp.copy()
        for position, factor in self.factors_mandel_to_voigt[quantity]:
            mandel[position] = inp[position] * 1.0 / factor
        return mandel

    def _voigt_to_mandel6_stress(self, inp):
        return self._voigt_to_mandel6(inp=inp, quantity="stress")

    def _voigt_to_mandel6_strain(self, inp):
        return self._voigt_to_mandel6(inp=inp, quantity="strain")

    def _voigt_to_mandel6_stiffness(self, inp):
        return self._voigt_to_mandel6(inp=inp, quantity="stiffness")

    def _voigt_to_mandel6_compliance(self, inp):
        return self._voigt_to_mandel6(inp=inp, quantity="compliance")

    def _voigt_umat_2(self, inp):
        # Is explicit copy necessary? Yes!?
        inp[..., [3, 5]] = inp[..., [5, 3]]
        return inp

    def _voigt_umat_4(self, inp):
        # Is explicit copy necessary? Yes!?
        inp[..., [3, 5], :] = inp[..., [5, 3], :]
        inp[..., :, [3, 5]] = inp[..., :, [5, 3]]
        return inp

    def _voigt_to_vumat_reorder_2(self, inp):
        inp[..., [3, 4]] = inp[..., [4, 3]]
        inp[..., [3, 5]] = inp[..., [5, 3]]
        return inp

    def _voigt_to_vumat_reorder_4(self, inp):
        inp[..., [3, 4], :] = inp[..., [4, 3], :]
        inp[..., :, [3, 4]] = inp[..., :, [4, 3]]

        inp[..., [3, 5], :] = inp[..., [5, 3], :]
        inp[..., :, [3, 5]] = inp[..., :, [5, 3]]
        return inp

    def _vumat_to_voigt_reorder_2(self, inp):
        inp[..., [3, 4]] = inp[..., [4, 3]]
        inp[..., [4, 5]] = inp[..., [5, 4]]
        return inp

    def _vumat_to_voigt_reorder_4(self, inp):
        inp[..., [3, 4], :] = inp[..., [4, 3], :]
        inp[..., :, [3, 4]] = inp[..., :, [4, 3]]

        inp[..., [4, 5], :] = inp[..., [5, 4], :]
        inp[..., :, [4, 5]] = inp[..., :, [5, 4]]
        return inp

    def _copy_and_scale(self, inp, factors):
        new = inp.copy()
        for position, factor in factors:
            new[position] = inp[position] * factor
        return new

    def _voigt_to_vumat_2(self, inp, quantity):
        new = self._copy_and_scale(
            inp=inp, factors=self.factors_voigt_to_reordered_vumat[quantity]
        )
        return self._voigt_to_vumat_reorder_2(new)

    def _voigt_to_vumat_4(self, inp, quantity):
        new = self._copy_and_scale(
            inp=inp, factors=self.factors_voigt_to_reordered_vumat[quantity]
        )
        return self._voigt_to_vumat_reorder_4(new)

    def _voigt_to_vumat_stress(self, inp):
        return self._voigt_to_vumat_2(inp=inp, quantity="stress")

    def _voigt_to_vumat_strain(self, inp):
        return self._voigt_to_vumat_2(inp=inp, quantity="strain")

    def _voigt_to_vumat_stiffness(self, inp):
        return self._voigt_to_vumat_4(inp=inp, quantity="stiffness")

    def _voigt_to_vumat_compliance(self, inp):
        return self._voigt_to_vumat_4(inp=inp, quantity="compliance")

    def _vumat_to_voigt_2(self, inp, quantity):
        new = self._copy_and_scale(
            inp=inp, factors=self.factors_reordered_vumat_to_voigt[quantity]
        )
        return self._vumat_to_voigt_reorder_2(new)

    def _vumat_to_voigt_4(self, inp, quantity):
        new = self._copy_and_scale(
            inp=inp, factors=self.factors_reordered_vumat_to_voigt[quantity]
        )
        return self._vumat_to_voigt_reorder_4(new)

    def _vumat_to_voigt_stress(self, inp):
        return self._vumat_to_voigt_2(inp=inp, quantity="stress")

    def _vumat_to_voigt_strain(self, inp):
        return self._vumat_to_voigt_2(inp=inp, quantity="strain")

    def _vumat_to_voigt_stiffness(self, inp):
        return self._vumat_to_voigt_4(inp=inp, quantity="stiffness")

    def _vumat_to_voigt_compliance(self, inp):
        return self._vumat_to_voigt_4(inp=inp, quantity="compliance")

    def _voigt_to_abaqusMaterialElasticAnisotropic(self, inp):
        # Abaqus2019 scripting reference Material.Elastic
        shape = inp.shape[:-2] + (21,)
        out = np.zeros(shape, dtype=np.float64)
        for i, row in enumerate(self.map_voigt_to_abaqusMaterialElasticAnisotropic):
            out[..., i] = inp[..., row[0], row[1]]
        return out

    def _abaqusMaterialElasticAnisotropic_to_voigt(self, inp):
        shape = inp.shape[:-1] + (6, 6)
        out = np.zeros(shape, dtype=np.float64)
        for i, row in enumerate(self.map_voigt_to_abaqusMaterialElasticAnisotropic):
            out[..., row[0], row[1]] = inp[..., i]
            if row[0] != row[1]:
                out[..., row[1], row[0]] = inp[..., i]
        return out


class Components(np.ndarray):

    stored_meta_data = ["notation", "quantity"]

    def __new__(cls, input_array, notation=None, quantity=None):
        cls.converter = ExplicitConverter()
        # Input array is an already formed ndarray instance
        # Cast to be our class type
        obj = np.asarray(input_array).view(cls)

        obj.notation = notation
        obj.quantity = quantity

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        else:
            self.copy_meta_info(new=self, old=obj)

    def copy_meta_info(self, new, old):
        for info in self.stored_meta_data:
            setattr(new, info, getattr(old, info, None))
        return new

    def wrapped(self, func):
        @functools.wraps(func)
        def wrapper_wrap_converter(target):
            new = func(
                inp=self,
                source=self.notation,
                quantity=self.quantity,
                # To support Python 2.7 without extended unpacking
                # https://stackoverflow.com/q/10792970/8935243
                target=target,
                # *args,
                # **kwargs,
            )
            new = Components(new)
            new.copy_meta_info(new=new, old=self)
            new.notation = target
            return new

        return wrapper_wrap_converter

    def to_tensor(self):
        return self.wrapped(self.converter.convert)(target="tensor")

    def to_mandel6(self):
        return self.wrapped(self.converter.convert)(target="mandel6")

    def to_mandel9(self):
        return self.wrapped(self.converter.convert)(target="mandel9")

    def to_voigt(self):
        return self.wrapped(self.converter.convert)(target="voigt")

    def to_umat(self):
        return self.wrapped(self.converter.convert)(target="umat")

    def to_vumat(self):
        return self.wrapped(self.converter.convert)(target="vumat")

    def to_abaqusMaterialAnisotropic(self):
        return self.wrapped(self.converter.convert)(target="abaqusMaterialAnisotropic")
