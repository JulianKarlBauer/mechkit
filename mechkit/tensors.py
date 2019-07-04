#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Basic tensors
'''

import numpy as np

DTYPE = 'float64'

###################################################################
# Basic tensors

I2 = np.eye(3, dtype=DTYPE)

I4 = np.einsum('ik, lj -> ijkl', I2, I2)

I4s = 0.5 * (I4 + np.einsum('ijkl -> ijlk', I4))

I4a = 0.5 * (I4 - np.einsum('ijkl -> ijlk', I4))

P1 = 1./3. * np.einsum('ij, kl -> ijkl', I2, I2)

P2 = I4s - P1


def levi_civita_tensor():
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    return eijk


ricci = levi_civita_tensor()
