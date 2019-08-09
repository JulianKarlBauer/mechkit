#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Material'''

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
        self._names_aliases = self._define_names()
        self._dict_funcs_to_K_G = self._funcs_to_K_G()
        self._con = mechkit.notation.Converter()
        self._tensors = mechkit.tensors.Basic()

        self._kwargs = kwargs
        self._input_names = self._get_input_names_from_kwargs(**kwargs)

        self._check_nbr_input_names()
        self._get_K_G()
        self._set_scalar()
        self._set_stiffness()
        self._set_compliance()

    def _define_names(self, ):
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

    def _funcs_to_K_G(self, ):
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
        return funcs_dict

    def _get_input_names_from_kwargs(self, **kwargs):
        input_names = {}
        for key, val in kwargs.items():
            for name, aliases in self._names_aliases.items():
                if key.lower() in aliases:
                    input_names[name] = val
        return input_names

    def _check_nbr_input_names(self, ):
        if len(self._input_names) != 2:
            raise Ex(
                'Number of input parameters has to be 2.\n'
                'Note: Isotropic material is defined by 2 parameters.\n'
                'Identified input parameters are:{}'.format(self._input_names)
                )

    def _get_funcs(self, names):
        return self._dict_funcs_to_K_G[frozenset(names)]

    def _get_K_G(self, ):
        func_K, func_G = self._get_funcs(self._input_names.keys())
        self.K = func_K(**self._input_names)
        self.G = func_G(**self._input_names)

    def _set_scalar(self, ):
        KG = {'K': self.K, 'G': self.G}
        attributes = {
                'K':        self.K,
                'G':        self.G,
                'mu':       self.G,
                'E':        self._E_by_K_G(**KG),
                'la':       self._la_by_K_G(**KG),
                'nu':       self._nu_by_K_G(**KG),
                'poisson':  self._nu_by_K_G(**KG),
                }
        for key, val in attributes.items():
            setattr(self, key, val)

    def _set_stiffness(self, ):
        self.stiffness = \
            3.*self.K*self._tensors.P1 \
            + 2.*self.G*self._tensors.P2
        self.stiffness_mandel6 = self._con.to_mandel6(self.stiffness)

    def _set_compliance(self, ):
        self.compliance_mandel6 = np.linalg.inv(
                                self.stiffness_mandel6
                                )
        self.compliance = self._con.to_tensor(self.compliance_mandel6)

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








