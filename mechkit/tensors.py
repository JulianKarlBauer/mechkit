#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Common tensors in tensor notation
'''
import numpy as np


class Basic(object):
    r'''
    Basic tensors in tensor notations

    Attributes
    ----------
    I2 : np.array of shape (3, 3,)
        Identity on second order tensors

        .. math::
            \begin{align*}
                \mathbf{I}
                &=
                \delta_{ij}
                    \;
                    \mathbf{e}_{i}
                    \otimes
                    \mathbf{e}_{j}
            \end{align*}

    I4 : np.array of shape (3, 3, 3, 3,)
        Identity on fourth order tensors

        .. math::
            \begin{align*}
                \mathbb{I}
                &=
                \delta_{ik} \delta_{lj}
                    \;
                    \mathbf{e}_{i}
                    \otimes
                    \mathbf{e}_{j}
                    \otimes
                    \mathbf{e}_{k}
                    \otimes
                    \mathbf{e}_{l} \\
            \end{align*}

    I4s : np.array of shape (3, 3, 3, 3,)
        Identity on symmetric fourth order tensors

        .. math::
            \begin{align*}
                \mathbb{I}^{\text{S}}
                &=
                \frac{1}{2}
                \left(
                    \mathbb{I} + \mathbb{I}^{\text{T}_\text{R}}
                \right)     \\
                &=
                \frac{1}{2}
                \left(
                    \delta_{ik} \delta_{lj} + \delta_{il} \delta_{kj}
                \right)
                    \;
                    \mathbf{e}_{i}
                    \otimes
                    \mathbf{e}_{j}
                    \otimes
                    \mathbf{e}_{k}
                    \otimes
                    \mathbf{e}_{l} \\
            \end{align*}

    I4a : np.array of shape (3, 3, 3, 3,)
        Identity on asymmetric fourth order tensors

        .. math::
            \begin{align*}
                \mathbb{I}^{\text{S}}
                &=
                \frac{1}{2}
                \left(
                    \mathbb{I} - \mathbb{I}^{\text{T}_\text{R}}
                \right)     \\
                &=
                \frac{1}{2}
                \left(
                    \delta_{ik} \delta_{lj} - \delta_{il} \delta_{kj}
                \right)
                    \;
                    \mathbf{e}_{i}
                    \otimes
                    \mathbf{e}_{j}
                    \otimes
                    \mathbf{e}_{k}
                    \otimes
                    \mathbf{e}_{l} \\
            \end{align*}

    P1 : np.array of shape (3, 3, 3, 3,)
        First isotropic projector.
        Projecting second order tensor onto its spherical part

        .. math::
            \begin{align*}
                \mathbb{P}_{\text{1}}
                &=
                \frac{1}{3}
                \mathbf{I} \otimes \mathbf{I}   \\
                &=
                \frac{1}{3}
                \delta_{ij}
                \delta_{kl}
                    \;
                    \mathbf{e}_{i}
                    \otimes
                    \mathbf{e}_{j}
                    \otimes
                    \mathbf{e}_{k}
                    \otimes
                    \mathbf{e}_{l}
            \end{align*}

    P2 : np.array of shape (3, 3, 3, 3,)
        Second isotropic projector.
        Projecting second order tensor onto its symmetric deviatoric part

        .. math::
            \begin{align*}
                \mathbb{P}_{\text{2}}
                &=
                \mathbb{I}^{\text{S}}
                -
                \mathbb{P}_{\text{1}}
            \end{align*}

    '''
    def __init__(self,):

        self.DTYPE = 'float64'
        self.I2 = np.eye(3, dtype=self.DTYPE)

        self.I4 = np.einsum('ik, lj -> ijkl', self.I2, self.I2)

        self.I4s = 0.5 * (self.I4 + np.einsum('ijkl -> ijlk', self.I4))

        self.I4a = 0.5 * (self.I4 - np.einsum('ijkl -> ijlk', self.I4))

        self.P1 = 1./3. * np.einsum('ij, kl -> ijkl', self.I2, self.I2)

        self.P2 = self.I4s - self.P1

        self.ricci = self._levi_civita_tensor()

        I2 = self.I2
        self.I6s = 1./8. * (np.einsum('ms, np, qr ->mnpqrs', I2, I2, I2) +
                            np.einsum('ms, nq, pr ->mnpqrs', I2, I2, I2) +
                            np.einsum('mr, np, qs ->mnpqrs', I2, I2, I2) +
                            np.einsum('mr, nq, ps ->mnpqrs', I2, I2, I2) +
                            np.einsum('mp, nr, qs ->mnpqrs', I2, I2, I2) +
                            np.einsum('mp, ns, qr ->mnpqrs', I2, I2, I2) +
                            np.einsum('mq, nr, ps ->mnpqrs', I2, I2, I2) +
                            np.einsum('mq, ns, pr ->mnpqrs', I2, I2, I2)
                            )

    def _levi_civita_tensor(self,):
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        return eijk
