#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Material'''

import numbers
import numpy as np
import mechkit
from mechkit.utils import Ex


class Isotropic(object):
    '''Representation of isotropic homogeneous material,
    defined by exactly **two** parameters.

    Features:

        - Accepts most common descriptors as keyword arguments.
        - Covers most common attributes.
        - Includes stiffness and compliance

    Attributes: (Following table 'Conversion formulae' [wikipedia]_)

        - **E** : Young's modulus
        - **poisson**, **nu** : Poissons ratio
        - **K** : Bulk modulus
        - **G**, **mu** : Shear modulus
        - **la** : Lame's first parameter
        - **stiffness** : Stiffness in tensor notation
        - **stiffness_mandel6** : Stiffness in Mandel6 notation
        - **compliance** : Compliance in tensor notation
        - **compliance_mandel6** : Compliance in Mandel6 notation

    Examples
    --------
    >>> import numpy as np
    >>> import mechkit

    >>> mat = Isotropic(E=200000, nu=0.3)
    >>> mat = Isotropic(K=16e4, NU=0.3)

    >>> mat = Isotropic(la=11e4, G=77e3)
    >>> mat.E
    199294.11
    >>> mat.nu
    0.29
    >>> mat.poisson
    0.29
    >>> mat.K
    161333.3
    >>> mat.G
    77000.0
    >>> mat.la
    109999.9

    >>> mat.stiffness_mandel6
    [[264000. 110000. 110000.      0.      0.      0.]
     [110000. 264000. 110000.      0.      0.      0.]
     [110000. 110000. 264000.      0.      0.      0.]
     [     0.      0.      0. 154000.      0.      0.]
     [     0.      0.      0.      0. 154000.      0.]
     [     0.      0.      0.      0.      0. 154000.]]
    >>> mat.compliance_mandel6
    [[ 5.02e-06 -1.48e-06 -1.48e-06  0.00e+00  0.00e+00  0.00e+00]
     [-1.48e-06  5.02e-06 -1.48e-06  0.00e+00  0.00e+00  0.00e+00]
     [-1.48e-06 -1.48e-06  5.02e-06  0.00e+00  0.00e+00  0.00e+00]
     [ 0.00e+00 -0.00e+00 -0.00e+00  6.49e-06 -0.00e+00 -0.00e+00]
     [ 0.00e+00  0.00e+00  0.00e+00  0.00e+00  6.49e-06  0.00e+00]
     [ 0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  6.49e-06]]
    >>> mat.stiffness
    [[[[264000.      0.      0.]
       [     0. 110000.      0.]
       [     0.      0. 110000.]]
      [[     0.  77000.      0.]
       [ 77000.      0.      0.]
       [     0.     ...


    .. rubric:: References

    .. [wikipedia] https://en.wikipedia.org/wiki/Elastic_modulus

    '''
    def __init__(self, **kwargs):
        self._con = mechkit.notation.Converter()
        self._tensors = mechkit.tensors.Basic()

        self._kwargs_keys = self._get_kwargs_keys_from_kwargs(**kwargs)
        self._check_nbr_kwargs_keys()
        self._get_K_G()

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
            'K':    ['k', 'compression', 'compression_modulus'],
            'G':    ['g', 'mu', 'shear', 'shear_modulus'],
            'E':    ['e', 'young', 'youngs', 'youngs_modulus',
                     'elastic_modulus'],
            'la':   ['la', 'l', 'lam', 'lamb', 'lambd', 'lambdaa',
                     'firstlame', 'first_lame', 'lamefirst', 'lame_first',
                     'lame1', 'lame_1'],
            'nu':   ['nu', 'poisson', 'poissonration', 'poisson_ration',
                     'poissons', 'poissonsratio', 'poisson_ration'],
            }
        return names_aliases

    def _func_to_K_G(self, keywords):
        f = frozenset
        funcs_dict = {
            f(['K',  'E']):     [self._K_by_K,       self._G_by_K_E],
            f(['K',  'la']):    [self._K_by_K,       self._G_by_K_la],
            f(['K',  'G']):     [self._K_by_K,       self._G_by_G],
            f(['K',  'nu']):    [self._K_by_K,       self._G_by_K_nu],
            f(['E',  'la']):    [self._K_by_E_la,    self._G_by_E_la],
            f(['E',  'G']):     [self._K_by_E_G,     self._G_by_G],
            f(['E',  'nu']):    [self._K_by_E_nu,    self._G_by_E_nu],
            f(['la', 'G']):     [self._K_by_la_G,    self._G_by_G],
            f(['la', 'nu']):    [self._K_by_la_nu,   self._G_by_la_nu],
            f(['G',  'nu']):    [self._K_by_G_nu,    self._G_by_G],
            }
        return funcs_dict[frozenset(keywords)]

    def _get_kwargs_keys_from_kwargs(self, **kwargs):
        kwargs_keys = {}
        for key, val in kwargs.items():
            for name, aliases in self._get_names_aliases().items():
                if key.lower() in aliases:
                    kwargs_keys[name] = val
        return kwargs_keys

    def _check_nbr_kwargs_keys(self, ):
        if len(self._kwargs_keys) != 2:
            raise Ex(
                'Number of input parameters has to be 2.\n'
                'Note: Isotropic material is defined by 2 parameters.\n'
                'Identified input parameters are:{}'.format(
                                                    self._kwargs_keys
                                                    )
                )

    def _get_K_G(self, ):
        func_K, func_G = self._func_to_K_G(keywords=self._kwargs_keys.keys())
        self.K = func_K(**self._kwargs_keys)
        self.G = func_G(**self._kwargs_keys)

    def _R_by_E_la(self, E, la):
        return np.sqrt(E*E + 9.*la*la + 2.*E*la)

    def _K_by_K(self, **kwargs):
        return kwargs['K']

    def _K_by_E_la(self, E, la):
        R = self._R_by_E_la(E, la)
        return (E + 3.*la + R) / 6.

    def _K_by_E_G(self, E, G):
        return (E*G) / (3.*(3.*G - E))

    def _K_by_E_nu(self, E, nu):
        return E / (3. * (1. - 2.*nu))

    def _K_by_la_G(self, la, G):
        return la + 2./3. * G

    def _K_by_la_nu(self, la, nu):
        return (la * (1. + nu)) / (3.*nu)

    def _K_by_G_nu(self, G, nu):
        return (2.*G * (1. + nu)) / (3. * (1. - 2.*nu))

    def _G_by_G(self, **kwargs):
        return kwargs['G']

    def _G_by_K_E(self, K, E):
        return (3.*K*E) / (9.*K - E)

    def _G_by_K_la(self, K, la):
        return (3. * (K - la)) / 2.

    def _G_by_K_nu(self, K, nu):
        return (3.*K * (1. - 2.*nu)) / (2. * (1. + nu))

    def _G_by_E_la(self, E, la):
        R = self._R_by_E_la(E, la)
        return (E - 3.*la + R) / (4.)

    def _G_by_E_nu(self, E, nu):
        return E / (2. * (1. + nu))

    def _G_by_la_nu(self, la, nu):
        return (la * (1. - 2.*nu)) / (2.*nu)

    def _E_by_K_G(self, K, G):
        return (9.*K*G) / (3.*K + G)

    def _la_by_K_G(self, K, G):
        return K - 2./3. * G

    def _nu_by_K_G(self, K, G):
        return (3.*K - 2.*G) / (2. * (3.*K + G))

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
    def poisson(self, ):
        return self.nu

    @property
    def stiffness(self, ):
        return 3.*self.K*self._tensors.P1 + 2.*self.G*self._tensors.P2

    @property
    def stiffness_mandel6(self, ):
        return self._con.to_mandel6(self.stiffness)

    @property
    def compliance_mandel6(self, ):
        return np.linalg.inv(self.stiffness_mandel6)

    @property
    def compliance(self, ):
        return self._con.to_tensor(self.compliance_mandel6)


if __name__ == '__main__':

    np.set_printoptions(
            linewidth=140,
            precision=2,
            # suppress=False,
            )

    mat = Isotropic(E=200000, nu=0.3)
    mat = Isotropic(K=16e4, NU=0.3)
    mat = Isotropic(la=11e4, G=77e3)

    printQueue = [
            'mat.E',
            'mat.nu',
            'mat.poisson',
            'mat.K',
            'mat.G',
            'mat.la',
            'mat.stiffness_mandel6',
            'mat.compliance_mandel6',
            'mat.stiffness',
            'mat.compliance',
            ]
    for val in printQueue:
        print(val)
        print(eval(val), '\n')








