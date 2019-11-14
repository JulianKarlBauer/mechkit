#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Material'''

import numbers
import numpy as np
import mechkit
import warnings
from mechkit.utils import Ex


class Isotropic(object):
    r'''Representation of homogeneous isotropic material.

    Use cases:

        - Create an instance of this class and

            - use the instance as an container representing the material
            - access most common material parameters as attributes or dict-like
              (implementation of [wikipedia_conversion_table]_)
            - do arithemtic on eigenvalues using the operators
              +, -, * with numbers.

    Quickstart:

    .. code-block:: python

        # Create instance
        mat = mechkit.material.Isotropic(E=2e6, nu=0.3)

        # Use attributes
        G = mat.G
        stiffness = mat.stiffness_mandel6


    **Two** independent material parameters uniquely define an isotropic
    material [Betram2015]_ (chapter 4.1.2).
    Therefore, exactly two material parameters have to be passed to the
    constructor of this class.

    The following primary arguments and aliases are **case-insensitive**
    **keyword arguments** of the constructor:

        - **K** : Compression modulus

            *Aliases* : compression_modulus

        - **G** : Shear modulus

            *Aliases* : mu, shear_modulus, second_lame, lame_2,
            C44_voigt, C55_voigt, C66_voigt,
            C2323, C1313, C1212

        - **E** : Youngs modulus

            *Aliases* : youngs_modulus, elastic_modulus

        - **la**: First Lame parameter

            *Aliases* : lambd, first_lame, lame_1,
            C23_voigt, C13_voigt, C12_voigt,
            C2233, C1133, C1122,
            C3322, C3311, C2211

        - **nu**: Poission's ratio

            *Aliases* : poisson, poisson_ratio

        - **M** : Constrained modulus or P-wave modulus

            *Aliases* : p_wave_modulus, longitudinal_modulus,
            constrained modulus,
            C11_voigt, C22_voigt, C33_voigt,
            C1111, C2222, C3333

    with

        - C<ij>_voigt : Component of stiffness matrix in Voigt notation
          (See note below)
        - C<ijkl> : Component of stiffness in tensor notation
          (See note below)


    Attributes: **(** Accessible both as attributes and dict-like
    ``mat['E']`` **)**

        - **K** : Bulk modulus
        - **G**, **mu** : Shear modulus
        - **E** : Young's modulus
        - **nu**, **poisson** : Poissons ratio
        - **la** : Lame's first parameter

        - **stiffness** : Stiffness in tensor notation
        - **stiffness_mandel6** : Stiffness in Mandel6 notation
        - **stiffness_voigt** : Stiffness in Voigt notation

        - **compliance** : Compliance in tensor notation
        - **compliance_mandel6** : Compliance in Mandel6 notation
        - **compliance_voigt** : Compliance in Voigt notation

    Warning
    -------

        Using parameters **E** and **M** leads to an **ambiguity**
        as the poisson's ratio can be positive or negative.

        - Use **auxetic=False** if you expect a **positive** poissons ratio.
        - Use **auxetic=True** if you expect a **negative** poissons ratio.


    Note
    ----

        Definition of stiffness components:

        Tensor components: (See [Betram2015]_ page 99 for details.)

        .. math::
            \begin{align*}
                \mathbb{C}
                &=
                C_{ijkl}
                \;
                \mathbf{e}_{i}
                \otimes
                \mathbf{e}_{j}
                \otimes
                \mathbf{e}_{k}
                \otimes
                \mathbf{e}_{l}\\
            \end{align*}

        Matrix components:

        .. math::
            \begin{align*}
                \mathbb{C}
                &=
                \begin{bmatrix}
             C_{11}  & C_{12}       & C_{13} & C_{14} & C_{15} & C_{16} \\
                     & C_{22}       & C_{23} & C_{24} & C_{25} & C_{26} \\
                     &              & C_{33} & C_{34} & C_{35} & C_{36} \\
                     &              &        & C_{44} & C_{45} & C_{46} \\
                     & \text{sym}   &        &        & C_{55} & C_{56} \\
                     &              &        &        &        & C_{66}
                \end{bmatrix}_{[\text{Voigt}]}      \hspace{-10mm}
                \scriptsize{
                    \boldsymbol{V}_{\alpha} \otimes \boldsymbol{V}_{\beta}
                    }   \\
                &=
                \begin{bmatrix}
             C_{11}  & C_{12}       & C_{13} & \sqrt{2}C_{14} & \sqrt{2}C_{15} & \sqrt{2}C_{16} \\
                     & C_{22}       & C_{23} & \sqrt{2}C_{24} & \sqrt{2}C_{25} & \sqrt{2}C_{26} \\
                     &              & C_{33} & \sqrt{2}C_{34} & \sqrt{2}C_{35} & \sqrt{2}C_{36} \\
                     &              &        & 2C_{44} & 2C_{45} & 2C_{46} \\
                     & \text{sym}   &        &         & 2C_{55} & 2C_{56} \\
                     &              &        &         &         & 2C_{66}
                \end{bmatrix}_{[\text{Mandel6}]}    \hspace{-15mm}
                \scriptsize{
                    \boldsymbol{B}_{\alpha} \otimes \boldsymbol{B}_{\beta}
                    }
            \end{align*}

        with

        - :math:`\boldsymbol{B}_{\alpha}` : Base dyad of Mandel6 notation
          (See :mod:`mechkit.notation`)
        - :math:`\boldsymbol{V}_{\alpha}` : Base dyad of Voigt notation
          (See [csmbrannonMandel]_)


    Note
    ----

        Isotropic linear elasticity:

        .. math::
            \begin{align*}
                \boldsymbol{\sigma}
                &=
                \mathbb{C}
                \left[
                    \boldsymbol{\varepsilon}
                \right]         \\
                &=
                \left(
                3 K \mathbb{P}_{\text{1}}
                +
                2 G \mathbb{P}_{\text{2}}
                \right)
                \left[
                    \boldsymbol{\varepsilon}
                \right]         \\
                &=
                \left(
                2 \mu \mathbb{I}^{\text{S}}
                +
                \lambda \mathbf{I} \otimes \mathbf{I}
                \right)
                \left[
                    \boldsymbol{\varepsilon}
                \right]
            \end{align*}

        with (See :class:`mechkit.tensors.Basic` for details and definitions)

        .. math::
            \begin{alignat}{2}
                \mathbb{I}^{\text{S}}
                &=
                \begin{bmatrix}
              1 & 0 & 0 & 0 & 0 & 0 \\
              0 & 1 & 0 & 0 & 0 & 0 \\
              0 & 0 & 1 & 0 & 0 & 0 \\
              0 & 0 & 0 & \frac{1}{2} & 0 & 0 \\
              0 & 0 & 0 & 0 & \frac{1}{2} & 0 \\
              0 & 0 & 0 & 0 & 0 & \frac{1}{2}
                \end{bmatrix}_{[\text{Voigt}]}      \hspace{-10mm}
                \scriptsize{
                    \boldsymbol{V}_{\alpha} \otimes \boldsymbol{V}_{\beta}
                    }
            &=
                \begin{bmatrix}
              1 & 0 & 0 & 0 & 0 & 0 \\
              0 & 1 & 0 & 0 & 0 & 0 \\
              0 & 0 & 1 & 0 & 0 & 0 \\
              0 & 0 & 0 & 1 & 0 & 0 \\
              0 & 0 & 0 & 0 & 1 & 0 \\
              0 & 0 & 0 & 0 & 0 & 1 \\
                \end{bmatrix}_{[\text{Mandel6}]}    \hspace{-15mm}
                \scriptsize{
                    \boldsymbol{B}_{\alpha} \otimes \boldsymbol{B}_{\beta}
                    }   \\
            \mathbf{I} \otimes \mathbf{I}
            &=
                \begin{bmatrix}
              1 & 1 & 1 & 0 & 0 & 0 \\
              1 & 1 & 1 & 0 & 0 & 0 \\
              1 & 1 & 1 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0 & 0 & 0 \\
                \end{bmatrix}_{[\text{Voigt}]}      \hspace{-10mm}
                \scriptsize{
                    \boldsymbol{V}_{\alpha} \otimes \boldsymbol{V}_{\beta}
                    }
            &=
                \begin{bmatrix}
              1 & 1 & 1 & 0 & 0 & 0 \\
              1 & 1 & 1 & 0 & 0 & 0 \\
              1 & 1 & 1 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0 & 0 & 0 \\
              0 & 0 & 0 & 0 & 0 & 0 \\
                \end{bmatrix}_{[\text{Mandel6}]}    \hspace{-15mm}
                \scriptsize{
                    \boldsymbol{B}_{\alpha} \otimes \boldsymbol{B}_{\beta}
                    }
            \end{alignat}

        Therefore, with
        :math:`\mu = G` and
        :math:`M = 2G + \lambda`
        the isotropic stiffness is

        .. math::
            \begin{align*}
                \mathbb{C}_{\text{isotropic}}
                &=
                \begin{bmatrix}
                    C_{11} & C_{12} & C_{13} & 0 & 0 & 0 \\
                    C_{12} & C_{22} & C_{23} & 0 & 0 & 0 \\
                    C_{13} & C_{23} & C_{33} & 0 & 0 & 0 \\
                    0 & 0 & 0 & C_{44} & 0 & 0 \\
                    0 & 0 & 0 & 0 & C_{55} & 0 \\
                    0 & 0 & 0 & 0 & 0 & C_{66}
                \end{bmatrix}_{[\text{Voigt}]}      \hspace{-10mm}
                \scriptsize{
                    \boldsymbol{V}_{\alpha} \otimes \boldsymbol{V}_{\beta}
                    }   \\
                &=
                \begin{bmatrix}
                    M & \lambda & \lambda & 0 & 0 & 0 \\
                    \lambda & M & \lambda & 0 & 0 & 0 \\
                    \lambda & \lambda & M & 0 & 0 & 0 \\
                    0 & 0 & 0 & G & 0 & 0 \\
                    0 & 0 & 0 & 0 & G & 0 \\
                    0 & 0 & 0 & 0 & 0 & G
                \end{bmatrix}_{[\text{Voigt}]}      \hspace{-10mm}
                \scriptsize{
                    \boldsymbol{V}_{\alpha} \otimes \boldsymbol{V}_{\beta}
                    }   \\
                &=
                \begin{bmatrix}
                    C_{11} & C_{12} & C_{13} & 0 & 0 & 0 \\
                    C_{12} & C_{22} & C_{23} & 0 & 0 & 0 \\
                    C_{13} & C_{23} & C_{33} & 0 & 0 & 0 \\
                    0 & 0 & 0 & 2C_{44} & 0 & 0 \\
                    0 & 0 & 0 & 0 & 2C_{55} & 0 \\
                    0 & 0 & 0 & 0 & 0 & 2C_{66}
                \end{bmatrix}_{[\text{Mandel6}]}    \hspace{-15mm}
                \scriptsize{
                    \boldsymbol{B}_{\alpha} \otimes \boldsymbol{B}_{\beta}
                    }  \\
                &=
                \begin{bmatrix}
                    M & \lambda & \lambda & 0 & 0 & 0 \\
                    \lambda & M & \lambda & 0 & 0 & 0 \\
                    \lambda & \lambda & M & 0 & 0 & 0 \\
                    0 & 0 & 0 & 2G & 0 & 0 \\
                    0 & 0 & 0 & 0 & 2G & 0 \\
                    0 & 0 & 0 & 0 & 0 & 2G
                \end{bmatrix}_{[\text{Mandel6}]}    \hspace{-15mm}
                \scriptsize{
                    \boldsymbol{B}_{\alpha} \otimes \boldsymbol{B}_{\beta}
                    }
            \end{align*}

    Examples
    --------
    >>> import mechkit

    >>> # Create instance
    >>> mat = mechkit.material.Isotropic(E=2e6, nu=0.3)
    >>> mat = mechkit.material.Isotropic(E=2e6, K=1e6)

    >>> # Access attributes
    >>> mat.G
    857142
    >>> mat['E']
    2000000

    >>> # More examples
    >>> mat1 = mechkit.material.Isotropic(M=15, G=5)
    >>> mat2 = mechkit.material.Isotropic(C11_voigt=20, C44_voigt=5)
    >>> mat1.stiffness_voigt
    [[15.  5.  5.  0.  0.  0.]
     [ 5. 15.  5.  0.  0.  0.]
     [ 5.  5. 15.  0.  0.  0.]
     [ 0.  0.  0.  5.  0.  0.]
     [ 0.  0.  0.  0.  5.  0.]
     [ 0.  0.  0.  0.  0.  5.]]
    >>> mat2['stiffness_voigt']
    [[20. 10. 10.  0.  0.  0.]
     [10. 20. 10.  0.  0.  0.]
     [10. 10. 20.  0.  0.  0.]
     [ 0.  0.  0.  5.  0.  0.]
     [ 0.  0.  0.  0.  5.  0.]
     [ 0.  0.  0.  0.  0.  5.]]
    >>> (0.5*mat1 + 0.5*mat2)['stiffness_voigt']
    [[17.5  7.5  7.5  0.   0.   0. ]
     [ 7.5 17.5  7.5  0.   0.   0. ]
     [ 7.5  7.5 17.5  0.   0.   0. ]
     [ 0.   0.   0.   5.   0.   0. ]
     [ 0.   0.   0.   0.   5.   0. ]
     [ 0.   0.   0.   0.   0.   5. ]]
    >>> mat1['stiffness_mandel6']
    [[15.  5.  5.  0.  0.  0.]
     [ 5. 15.  5.  0.  0.  0.]
     [ 5.  5. 15.  0.  0.  0.]
     [ 0.  0.  0. 10.  0.  0.]
     [ 0.  0.  0.  0. 10.  0.]
     [ 0.  0.  0.  0.  0. 10.]]
    >>> mat1['compliance_mandel6']
    [[ 0.08 -0.02 -0.02  0.    0.    0.  ]
     [-0.02  0.08 -0.02  0.    0.    0.  ]
     [-0.02 -0.02  0.08  0.    0.    0.  ]
     [ 0.   -0.   -0.    0.1  -0.   -0.  ]
     [ 0.    0.    0.    0.    0.1   0.  ]
     [ 0.    0.    0.    0.    0.    0.1 ]]


    .. rubric:: References

    .. [Betram2015] Bertram, A., & Glüge, R. (2015).
        Solid mechanics. Springer Int. Publ.

    .. [wikipedia_conversion_table]
       https://en.wikipedia.org/wiki/Template:Elastic_moduli

    '''
    def __init__(self, auxetic=False, **kwargs):
        self._con = mechkit.notation.VoigtConverter(silent=True)
        self._tensors = mechkit.tensors.Basic()
        self.auxetic = auxetic

        self._useful_kwargs = self._get_useful_kwargs_from_kwargs(**kwargs)
        self._check_nbr_useful_kwargs(**kwargs)
        self._get_K_G()
        self._check_positive_definiteness()

    def __getitem__(self, key):
        '''Make attributes accessible dict-like.'''
        return getattr(self, key)

    def __add__(self, other):
        K = self.K + other.K
        G = self.G + other.G
        return Isotropic(K=K, G=G)

    def __sub__(self, other):
        K = self.K - other.K
        G = self.G - other.G
        return Isotropic(K=K, G=G)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            K = other * self.K
            G = other * self.G
        else:
            raise NotImplementedError('Multiply only with numbers.')
        return Isotropic(K=K, G=G)

    def __rmul__(self, other):
        return self.__mul__(other)

    def _get_names_aliases(self, ):
        names_aliases = {
            'K':    ['k', 'compression_modulus', ],
            'G':    ['g', 'mu', 'shear_modulus',
                     'second_lame', 'lame_2',
                     'c44_voigt', 'c55_voigt', 'c66_voigt',
                     'c2323', 'c1313', 'c1212'],
            'E':    ['e', 'youngs_modulus', 'elastic_modulus', ],
            'nu':   ['nu', 'poisson', 'poisson_ratio', ],
            'la':   ['la', 'lambd', 'first_lame', 'lame_1',
                     'c23_voigt', 'c13_voigt', 'c12_voigt',
                     'c32_voigt', 'c31_voigt', 'c21_voigt',
                     'c2233', 'c1133', 'c1122',
                     'c3322', 'c3311', 'c2211', ],
            'M':    ['m', 'p_wave_modulus', 'longitudinal_modulus',
                     'constrained modulus',
                     'c11_voigt', 'c22_voigt', 'c33_voigt',
                     'c1111', 'c2222', 'c3333', ],
            }
        return names_aliases

    def _func_to_K_G(self, keywords):
        f = frozenset
        funcs_dict = {
            f(['K',  'E']):     [self._K_by_K,       self._G_by_K_E],
            f(['K',  'la']):    [self._K_by_K,       self._G_by_K_la],
            f(['K',  'G']):     [self._K_by_K,       self._G_by_G],
            f(['K',  'nu']):    [self._K_by_K,       self._G_by_K_nu],
            f(['K',  'M']):     [self._K_by_K,       self._G_by_K_M],
            f(['E',  'la']):    [self._K_by_E_la,    self._G_by_E_la],
            f(['E',  'G']):     [self._K_by_E_G,     self._G_by_G],
            f(['E',  'nu']):    [self._K_by_E_nu,    self._G_by_E_nu],
            f(['E',  'M']):     [self._K_by_E_M,     self._G_by_E_M],
            f(['la', 'G']):     [self._K_by_la_G,    self._G_by_G],
            f(['la', 'nu']):    [self._K_by_la_nu,   self._G_by_la_nu],
            f(['la', 'M']):     [self._K_by_la_M,    self._G_by_la_M],
            f(['G',  'nu']):    [self._K_by_G_nu,    self._G_by_G],
            f(['G',  'M']):     [self._K_by_G_M,     self._G_by_G],
            f(['nu',  'M']):    [self._K_by_nu_M,    self._G_by_nu_M],
            }
        return funcs_dict[frozenset(keywords)]

    def _get_useful_kwargs_from_kwargs(self, **kwargs):
        useful = {}
        for key, val in kwargs.items():
            for name, aliases in self._get_names_aliases().items():
                if key.lower() in aliases:
                    if name not in useful:
                        useful[name] = val
                    else:
                        raise Ex(
                            ('Redundant input for primary parameter {name}\n'
                             'Failed to use \n{key}={val}\nbecause {name} '
                             'is already assigned the value {useful}\n'
                             'Given arguments are:{kwargs}\n'
                             ).format(
                                name=name,
                                key=key,
                                val=val,
                                useful=useful[name],
                                kwargs=kwargs,
                                )
                            )
        return useful

    def _check_nbr_useful_kwargs(self, **kwargs):
        if len(self._useful_kwargs) != 2:
            raise Ex(
                ('Number of input parameters has to be 2.\n'
                 'Note: Isotropic material is defined by 2 parameters.\n'
                 'Given arguments are:{}\n'
                 'Identified primary input parameters are:{}\n').format(
                                                    kwargs,
                                                    self._useful_kwargs
                                                    )
                )

    def _check_positive_definiteness(self, ):
        if not ((self.K >= 0.) and (self.G >= 0.)):
            raise Ex(
                'Negative K or G.\n'
                'K and G of positiv definit isotropic material'
                'have to be positive. \nK={} G={}'.format(self.K, self.G)
                )

    def _get_K_G(self, ):
        func_K, func_G = self._func_to_K_G(keywords=self._useful_kwargs.keys())
        self.K = func_K(**self._useful_kwargs)
        self.G = func_G(**self._useful_kwargs)

    def _R_by_E_la(self, E, la):
        return np.sqrt(E*E + 9.*la*la + 2.*E*la)

    def _S_by_E_M(self, E, M):
        warnings.warn(
            message=(
             "Using parameters 'E' and 'M' leads to an ambiguity.\n"
             "Use 'auxetic=False' if you expect a positive poissons ratio.\n"
             "Use 'auxetic=True' if you expect a negative poissons ratio."),
            category=UserWarning,
            )
        S = np.sqrt(E**2 + 9.*M**2 - 10.*E*M)
        return S if not self.auxetic else -S

    def _K_by_K(self, **kwargs):
        return kwargs['K']

    def _K_by_E_la(self, E, la):
        R = self._R_by_E_la(E, la)
        return (E + 3.*la + R) / 6.

    def _K_by_E_G(self, E, G):
        return (E*G) / (3.*(3.*G - E))

    def _K_by_E_nu(self, E, nu):
        return E / (3. * (1. - 2.*nu))

    def _K_by_E_M(self, E, M):
        return (3.*M - E + self._S_by_E_M(E=E, M=M)) / 6.

    def _K_by_la_G(self, la, G):
        return la + 2./3. * G

    def _K_by_la_nu(self, la, nu):
        return (la * (1. + nu)) / (3.*nu)

    def _K_by_la_M(self, la, M):
        return (M + 2.*la) / 3.

    def _K_by_G_nu(self, G, nu):
        return (2.*G * (1. + nu)) / (3. * (1. - 2.*nu))

    def _K_by_G_M(self, G, M):
        return M - (4.*G)/3

    def _K_by_nu_M(self, nu, M):
        return (M*(1.+nu)) / (3.*(1-nu))

    def _G_by_G(self, **kwargs):
        return kwargs['G']

    def _G_by_K_E(self, K, E):
        return (3.*K*E) / (9.*K - E)

    def _G_by_K_la(self, K, la):
        return (3. * (K - la)) / 2.

    def _G_by_K_nu(self, K, nu):
        return (3.*K * (1. - 2.*nu)) / (2. * (1. + nu))

    def _G_by_K_M(self, K, M):
        return 3.*(M - K) / 4.

    def _G_by_E_la(self, E, la):
        R = self._R_by_E_la(E, la)
        return (E - 3.*la + R) / (4.)

    def _G_by_E_nu(self, E, nu):
        return E / (2. * (1. + nu))

    def _G_by_E_M(self, E, M):
        return (3.*M + E - self._S_by_E_M(E=E, M=M)) / 8.

    def _G_by_la_nu(self, la, nu):
        return (la * (1. - 2.*nu)) / (2.*nu)

    def _G_by_la_M(self, la, M):
        return (M - la) / 2.

    def _G_by_nu_M(self, nu, M):
        return (M*(1 - 2.*nu)) / (2.*(1 - nu))

    def _E_by_K_G(self, K, G):
        return (9.*K*G) / (3.*K + G)

    def _la_by_K_G(self, K, G):
        return K - 2./3. * G

    def _nu_by_K_G(self, K, G):
        return (3.*K - 2.*G) / (2. * (3.*K + G))

    def _M_by_K_G(self, K, G):
        return K + (4.*G)/3.

    @property
    def mu(self, ):
        return self.G

    @property
    def E(self, ):
        return self._E_by_K_G(K=self.K, G=self.G)

    @property
    def la(self, ):
        return self._la_by_K_G(K=self.K, G=self.G)

    @property
    def nu(self, ):
        return self._nu_by_K_G(K=self.K, G=self.G)

    @property
    def M(self, ):
        return self._M_by_K_G(K=self.K, G=self.G)

    @property
    def poisson(self, ):
        return self.nu

    @property
    def stiffness(self, ):
        return 3.*self.K*self._tensors.P1 + 2.*self.G*self._tensors.P2

    @property
    def stiffness_mandel6(self, ):
        return self._con.to_mandel6(self.stiffness)

    @property
    def stiffness_voigt(self, ):
        return self._con.mandel6_to_voigt(
                    self.stiffness_mandel6,
                    voigt_type='stiffness',
                    )

    @property
    def compliance_mandel6(self, ):
        return np.linalg.inv(self.stiffness_mandel6)

    @property
    def compliance(self, ):
        return self._con.to_tensor(self.compliance_mandel6)

    @property
    def compliance_voigt(self, ):
        return self._con.mandel6_to_voigt(
                    self.compliance_mandel6,
                    voigt_type='compliance',
                    )


if __name__ == '__main__':

    np.set_printoptions(
            linewidth=140,
            precision=2,
            # suppress=False,
            )

    mat = mechkit.material.Isotropic(E=2e6, nu=0.3)
    mat = mechkit.material.Isotropic(E=2e6, K=1e6)
    mat1 = mechkit.material.Isotropic(M=15, G=5)
    mat2 = mechkit.material.Isotropic(C11_voigt=20, C44_voigt=5)

    printQueue = [
            "mat.G",
            "mat['E']",
            "mat1['stiffness_voigt']",
            "mat2['stiffness_voigt']",
            "(0.5*mat1 + 0.5*mat2)['stiffness_voigt']",
            "mat1['stiffness_mandel6']",
            "mat1['compliance_mandel6']",
            ]
    for val in printQueue:
        print(val)
        print(eval(val), '\n')








