#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Common tensors in tensor notation
'''
import numpy as np


class basic(object):
    '''Basic tensors: Identities, isotropic projectors...'''
    def __init__(self,):

        self.DTYPE = 'float64'
        self.I2 = np.eye(3, dtype=self.DTYPE)

        self.I4 = np.einsum('ik, lj -> ijkl', self.I2, self.I2)

        self.I4s = 0.5 * (self.I4 + np.einsum('ijkl -> ijlk', self.I4))

        self.I4a = 0.5 * (self.I4 - np.einsum('ijkl -> ijlk', self.I4))

        self.P1 = 1./3. * np.einsum('ij, kl -> ijkl', self.I2, self.I2)

        self.P2 = self.I4s - self.P1

        self.ricci = self.levi_civita_tensor()

    def levi_civita_tensor(self,):
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        return eijk
