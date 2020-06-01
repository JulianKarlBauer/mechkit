#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Material'''

import numbers
import numpy as np
import mechkit
import warnings
from mechkit.utils import Ex


class AbstractMaterial(object):
    def __init__(self, **kwargs):
        self._con = mechkit.notation.VoigtConverter(silent=True)

    def __getitem__(self, key):
        '''Make attributes accessible dict-like.'''
        return getattr(self, key)

    def _get_useful_kwargs_from_kwargs(self, **kwargs):
        names_aliases = self._get_names_aliases()

        useful = {}
        for key, val in kwargs.items():
            for name, aliases in names_aliases.items():
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
        if len(kwargs) != len(useful):
            raise Ex(
                    ('Not all keyword arguments are identified as material '
                     'parameters.\n'
                     'Identified material parameters: {useful}\n'
                     'Given kwargs are: {kwargs}'
                     ).format(
                        useful=useful,
                        kwargs=kwargs,
                        )
                )
        return useful

    def _check_nbr_useful_kwargs(self, **kwargs):
        if len(self._useful_kwargs) != self._nbr_useful_kwargs:
            raise Ex(
                ('Number of input parameters has to be {nbr}.\n'
                 'Note: {mat} is defined by {nbr} parameters.\n'
                 'Given arguments are:{kwargs}\n'
                 'Identified primary input parameters are:{useful}\n').format(
                                                    kwargs=kwargs,
                                                    useful=self._useful_kwargs,
                                                    nbr=self._nbr_useful_kwargs,
                                                    mat=type(self)
                                                    )
                )

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


class Isotropic(AbstractMaterial):
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

    with (See :ref:`DefinitionStiffnessComponents`)

        - C<ij>_voigt : Component of stiffness matrix in Voigt notation
        - C<ijkl> : Component of stiffness in tensor notation

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
        super(type(self), self).__init__()

        self._tensors = mechkit.tensors.Basic()
        self._nbr_useful_kwargs = 2
        self.auxetic = auxetic

        self._useful_kwargs = self._get_useful_kwargs_from_kwargs(**kwargs)
        self._check_nbr_useful_kwargs(**kwargs)
        self._get_K_G()
        self._check_positive_definiteness()

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
            'nu':   ['nu', 'poisson', 'poisson_ratio', 'v'],
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

    def _check_positive_definiteness(self, ):
        if not ((self.K >= 0.) and (self.G >= 0.)):
            raise Ex(
                'Negative K or G.\n'
                'K and G of positiv definit isotropic material '
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


class Orthotropic():
    r'''Representation of homogeneous orthotropic material.

    **Nine** independent material parameters uniquely define an orthotropic
    material [Betram2015]_ (chapter 4.1.2), aligned with the coordinate axes.

    See definitions of :ref:`EngineeringConstants`.

    Attributes:

    - Keyword arguments

    - **stiffness** : Stiffness in tensor notation
    - **stiffness_mandel6** : Stiffness in Mandel6 notation
    - **stiffness_voigt** : Stiffness in Voigt notation

    - **compliance** : Compliance in tensor notation
    - **compliance_mandel6** : Compliance in Mandel6 notation
    - **compliance_voigt** : Compliance in Voigt notation


    Compliance

        .. math::
            \begin{align*}
                \mathbb{C}^{-1}_{\text{orthotropic}}
                &=
                \begin{bmatrix}
                    \frac{1}{E_1}   & -\frac{\nu_{21}}{E_2}     & -\frac{\nu_{31}}{E_3} & 0 & 0 & 0 \\
                    -\frac{\nu_{12}}{E_1}   & \frac{1}{E_2}     & -\frac{\nu_{32}}{E_3} & 0 & 0 & 0 \\
                    -\frac{\nu_{13}}{E_1}   & -\frac{\nu_{23}}{E_2} & \frac{1}{E_3}     & 0 & 0 & 0 \\
                    0 & 0 & 0 & \frac{1}{G_{23}} & 0 & 0 \\
                    0 & 0 & 0 & 0 & \frac{1}{G_{31}} & 0 \\
                    0 & 0 & 0 & 0 & 0 & \frac{1}{G_{12}}
                \end{bmatrix}_{[\text{Voigt}]}  \\
                &=
                \begin{bmatrix}
                    \frac{1}{E_1}   & -\frac{\nu_{12}}{E_1}     & -\frac{\nu_{13}}{E_1} & 0 & 0 & 0 \\
                                    & \frac{1}{E_2}     & -\frac{\nu_{23}}{E_2} & 0 & 0 & 0 \\
                                    &                   &\frac{1}{E_3}     & 0 & 0 & 0 \\
                    & & & \frac{1}{G_{23}} & 0 & 0 \\
                    & sym & & & \frac{1}{G_{31}} & 0 \\
                    & & & & & \frac{1}{G_{12}}
                \end{bmatrix}_{[\text{Voigt}]}
            \end{align*}

    '''

    def __init__(self, E1, E2, E3, nu12, nu13, nu23, G12, G13, G23):

        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.nu12 = nu12
        self.nu13 = nu13
        self.nu23 = nu23
        self.G12 = G12
        self.G13 = G13
        self.G23 = G23

        S12 = -nu12 / E1
        S13 = -nu13 / E1
        S23 = -nu23 / E2
        self.compliance_voigt = np.array(
                [
                    [1./E1, S12,    S13,    0,      0,      0],
                    [S12,   1./E2,  S23,    0,      0,      0],
                    [S13,   S23,    1./E3,  0,      0,      0],
                    [0,     0,      0,      1./G23, 0,      0],
                    [0,     0,      0,      0,      1./G13, 0],
                    [0,     0,      0,      0,      0,      1./G12],
                ],
                dtype='float64',
                )

        self._con = mechkit.notation.VoigtConverter(silent=True)

    @property
    def compliance_mandel6(self, ):
        return self._con.voigt_to_mandel6(
                   self.compliance_voigt,
                   voigt_type='compliance',
                   )

    @property
    def compliance(self, ):
        return self._con.to_tensor(self.compliance_mandel6)

    @property
    def stiffness_mandel6(self, ):
        return np.linalg.inv(self.compliance_mandel6)

    @property
    def stiffness(self, ):
        return self._con.to_tensor(self.stiffness_mandel6)

    @property
    def stiffness_voigt(self, ):
        return self._con.mandel6_to_voigt(
                   self.stiffness_mandel6,
                   voigt_type='stiffness',
                   )


class TransversalIsotropic(AbstractMaterial):
    r'''Representation of homogeneous transversal isotropic material.

    Quickstart:

    .. code-block:: python

        # Create instance
        mat = mechkit.material.TransversalIsotropic(
            E_l=100.0, E_t=20.0, nu_lt=0.3, G_lt=10.0, G_tt=7.0,
            principal_axis=[1, 1, 0]
        )

        # Use attributes
        stiffness = mat.stiffness_mandel6

    **Five** independent material parameters uniquely define an isotropic
    material [Betram2015]_ (chapter 4.1.2).
    Therefore, exactly five material parameters have to be passed to the
    constructor of this class.

    See definitions of :ref:`EngineeringConstants`.

    Coordinate-free indices are used

        - *l* : longitudinal, i.e. in direction of principal axis
        - *t* : transversal, i.e. perpendicular to principal axis

    and the direction of the principal axis can be given in vector format as
    **principal_axis**. The default principal axis is the x-axis.
    The vector does not have to be normalized.

    Valid **case-insensitive** keyword arguments and aliases of the constructor are
    (Format: - **keyword arguments** : Aliases)

        - **E_l** : El
        - **E_t** : Et
        - **G_lt** : Glt
        - **G_tt** : Gtt
        - **nu_lt** : nult, v_lt, vlt
        - **nu_tl** : nutl, v_tl, vtl
        - **nu_tt** : nutt, v_tt, vtt

    Only four combinations of these arguments are valid:

        - **E_l**, **E_t**, **G_lt**, **G_tt**, **nu_lt**
        - **E_l**, **E_t**, **G_lt**, **G_tt**, **nu_tl**
        - **E_l**, **E_t**, **G_lt**, **nu_lt**, **nu_tt**
        - **E_l**, **E_t**, **G_lt**, **nu_tl**, **nu_tt**

    Attributes: **(** Accessible both as attributes and dict-like **)**

        - **E_l**, **E_t**, **G_lt**, **G_tt**, **nu_lt**, **nu_tl**, **nu_tt**

        - **stiffness** : Stiffness in tensor notation
        - **stiffness_mandel6** : Stiffness in Mandel6 notation
        - **stiffness_voigt** : Stiffness in Voigt notation

        - **compliance** : Compliance in tensor notation
        - **compliance_mandel6** : Compliance in Mandel6 notation
        - **compliance_voigt** : Compliance in Voigt notation

    Examples
    --------
    >>> import mechkit

    >>> # Create instance
    >>> mat = mechkit.material.TransversalIsotropic(
            E_l=100.0, E_t=20.0, nu_lt=0.3, G_lt=10.0, G_tt=7.0,
            principal_axis=[0, 1, 0]
        )

    >>> # Access attributes
    >>> mat.compliance_voigt
    [[ 0.05  -0.003 -0.021  0.     0.     0.   ]
     [-0.003  0.01  -0.003  0.     0.     0.   ]
     [-0.021 -0.003  0.05   0.     0.     0.   ]
     [ 0.    -0.    -0.     0.1   -0.    -0.   ]
     [ 0.     0.     0.     0.     0.143  0.   ]
     [ 0.     0.     0.     0.     0.     0.1  ]]


    >>> mat.stiffness_mandel6
    [[ 25.68  11.21  11.68   0.     0.     0.  ]
     [ 11.21 106.72  11.21   0.     0.     0.  ]
     [ 11.68  11.21  25.68   0.     0.     0.  ]
     [  0.     0.     0.    20.     0.     0.  ]
     [  0.     0.     0.     0.    14.     0.  ]
     [  0.     0.     0.     0.     0.    20.  ]]

    '''

    def __init__(self, principal_axis=[1, 0, 0], **kwargs):
        super(type(self), self).__init__()

        self.principal_axis = principal_axis
        self._nbr_useful_kwargs = 5
        self._default_principal_axis = [1, 0, 0]

        self._useful_kwargs = self._get_useful_kwargs_from_kwargs(**kwargs)
        self._check_nbr_useful_kwargs(**kwargs)
        self._get_primary_parameters()
        self.stiffness = Orthotropic(
                                    E1=self.E_l,
                                    E2=self.E_t,
                                    E3=self.E_t,
                                    nu12=self.nu_lt,
                                    nu13=self.nu_lt,
                                    nu23=self._nu_tt(),
                                    G12=self.G_lt,
                                    G13=self.G_lt,
                                    G23=self.G_tt,
                                    ).stiffness
        self._check_positive_definiteness()

        if self.principal_axis != self._default_principal_axis:
            self.stiffness = self._rotate_stiffness_into_principal_axis()

    def _get_names_aliases(self, ):
        '''Note: There are different definitions of poissons ratio.
        (VDI 2014 Blatt 3 page 14)
        In case of the Poisson’s ratios there are different ways
        for the indexing in international practice. In the
        guideline VDI 2014 the required two indices are uti-
        lized as follows: The 1 st index indicates the direction
        of the transverse contraction. The 2 nd index denotes
        the stress, which causes the contraction. As a conse-
        quence the Poisson’s ratios nu_tl is the larger and nu_lt
        the smaller one. (In the English literature the two indices related to the
        contraction and acting stress are
        used in the reverse sequence.)
        '''
        names_aliases = {
            'E_l':   ['e_l', 'el'],
            'E_t':   ['e_t', 'et'],
            'G_lt':  ['g_lt', 'glt'],
            'G_tt':  ['g_tt', 'gtt'],
            'nu_lt': ['nu_lt', 'nult', 'v_lt', 'vlt'],
            'nu_tl': ['nu_tl', 'nutl', 'v_tl', 'vtl'],
            'nu_tt': ['nu_tt', 'nutt', 'v_tt', 'vtt'],
            }
        return names_aliases

    def _raise_required(self, key):
        raise Ex(
                    (
                        'Parameter {key} is required.\n'
                        'Aliases are {aliases}.'
                    ).format(
                        key=key,
                        aliases=self._get_names_aliases[key],
                        )
                )

    def _raise_required_either_or(self, keys):
        raise Ex(
                    (
                        'Either {key0} or {key1} is required.\n'
                        'Aliases of {key0} are {aliases0}\n'
                        'Aliases of {key1} are {aliases1}\n'
                    ).format(
                        key0=keys[0],
                        key1=keys[1],
                        aliases0=self._get_names_aliases[keys[0]],
                        aliases1=self._get_names_aliases[keys[1]],
                        )
                )

    def _get_primary_parameters(self, ):
        useful = self._useful_kwargs

        for key in ['E_l', 'E_t', 'G_lt']:
            if key in useful:
                setattr(self, key, useful[key])
            else:
                self._raise_required(key=key)

        if 'nu_lt' in useful:
            self.nu_lt = useful['nu_lt']
        elif 'nu_tl' in useful:
            self.nu_lt = self._nu_lt(nu_tl=useful['nu_tl'])
        else:
            self._raise_required_either_or(keys=['nu_lt', 'nu_tl'])

        if 'G_tt' in useful:
            self.G_tt = useful['G_tt']
        elif 'nu_tt' in useful:
            self.G_tt = self._G_tt(nu_tt=useful['nu_tt'])
        else:
            self._raise_required_either_or(keys=['G_tt', 'nu_tt'])

    def _check_positive_definiteness(self, ):
        if not (0.0 < min(np.linalg.eigh(self.stiffness_mandel6)[0])):
            raise Ex(
                'Stiffness Mandel6 is not positive definite'
                )

    def _nu_lt(self, nu_tl):
        return nu_tl * self.E_l / self.E_t

    def _nu_tl(self, nu_lt):
        return nu_lt * self.E_t / self.E_l

    def _G_tt(self, nu_tt):
        return self.E_t / (2. * (1. + nu_tt))

    def _nu_tt(self, ):
        return self.E_t / (2. * self.G_tt) - 1.

    def _get_rotation_matrix(self, start_vector, end_vector):
        '''Thanks to https://math.stackexchange.com/a/2672702/694025'''

        a = np.array(start_vector, dtype=np.float64)
        b = np.array(end_vector, dtype=np.float64)

        # Reshape to get Matlab-like operations and normalize
        a = a.reshape(3, 1) / np.linalg.norm(a)
        b = b.reshape(3, 1) / np.linalg.norm(b)

        c = a + b
        return 2.0 * np.matmul(c, c.T) / np.matmul(c.T, c) - np.eye(3)

    def _rotate_stiffness_into_principal_axis(self, ):
        R = self._get_rotation_matrix(
                        start_vector=self._default_principal_axis,
                        end_vector=self.principal_axis,
                        )
        return np.einsum('ij, kl, mn, op, jlnp->ikmo', R, R, R, R, self.stiffness)

    @property
    def nu_tl(self, ):
        return self._nu_tl(nu_lt=self.nu_lt)

    @property
    def nu_tt(self, ):
        return self._nu_tt()


if __name__ == '__main__':

    np.set_printoptions(
            linewidth=140,
            precision=3,
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

    mat = mechkit.material.TransversalIsotropic(
        E_l=100.0, E_t=20.0, nu_lt=0.3, G_lt=10.0, G_tt=7.0, principal_axis=[0, 1, 0]
    )

    printQueue = [
            "mat.compliance_voigt",
            "mat.stiffness_mandel6",
            ]
    for val in printQueue:
        print(val)
        print(eval(val), '\n')









