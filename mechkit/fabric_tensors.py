#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''fabric tensors in Mandel6 notation
'''
import numpy as np
import mechkit


class Basic(object):
    r'''
    Fabric tensors of the first kind ([Kanatani1984]_) for special cases

    - iso
    - planar_iso_xy
    - planar_iso_xz
    - planar_iso_yz
    - ud_x
    - ud_y
    - ud_z

    .. rubric:: References

    .. [Kanatani1984] Ken-Ichi, K. (1984).
        Distribution of directional data and fabric tensors.
        International Journal of Engineering Science, 22(2), 149-164.

    Examples
    --------
    >>> import mechkit

    >>> N2 = mechkit.fabric_tensors.Basic()['N2']['iso']
    >>> N4 = mechkit.fabric_tensors.Basic()['N4']['iso']

    '''
    def __init__(self, ):
        self.N4 = {
            'iso':
                1./5. * np.array(
                 [[1.,      1./3.,     1./3.,     0.,     0.,     0., ],
                  [1./3.,   1.,        1./3.,     0.,     0.,     0., ],
                  [1./3.,   1./3.,     1.,        0.,     0.,     0., ],
                  [0.,      0.,        0.,        2./3.,  0.,     0., ],
                  [0.,      0.,        0.,        0.,     2./3.,  0., ],
                  [0.,      0.,        0.,        0.,     0.,     2./3., ],
                  ]),
            'planar_iso_xy':
                1./8. * np.array(
                     [[3.,      1.,      0.,       0.,     0.,     0., ],
                      [1.,      3.,      0.,       0.,     0.,     0., ],
                      [0.,      0.,      0.,       0.,     0.,     0., ],
                      [0.,      0.,      0.,       0.,     0.,     0., ],
                      [0.,      0.,      0.,       0.,     0.,     0., ],
                      [0.,      0.,      0.,       0.,     0.,     2., ],
                      ]),
            'planar_iso_xz':
                1./8. * np.array(
                     [[3.,      0.,      1.,       0.,     0.,     0., ],
                      [0.,      0.,      0.,       0.,     0.,     0., ],
                      [1.,      0.,      3.,       0.,     0.,     0., ],
                      [0.,      0.,      0.,       0.,     0.,     0., ],
                      [0.,      0.,      0.,       0.,     2.,     0., ],
                      [0.,      0.,      0.,       0.,     0.,     0., ],
                      ]),
            'planar_iso_yz':
                1./8. * np.array(
                     [[0.,      0.,      0.,       0.,     0.,     0., ],
                      [0.,      3.,      1.,       0.,     0.,     0., ],
                      [0.,      1.,      3.,       0.,     0.,     0., ],
                      [0.,      0.,      0.,       2.,     0.,     0., ],
                      [0.,      0.,      0.,       0.,     0.,     0., ],
                      [0.,      0.,      0.,       0.,     0.,     0., ],
                      ]),
            'ud_x':
                np.array(
                     [[1.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      ]),
            'ud_y':
                np.array(
                     [[0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      1.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      ]),
            'ud_z':
                np.array(
                     [[0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       1.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      [0.,      0.,       0.,       0.,     0.,     0., ],
                      ]),
            }
        con = mechkit.notation.Converter()
        I2 = con.to_mandel6(mechkit.tensors.Basic().I2)

        self.N2 = {direction: np.matmul(val, I2) for direction, val in self.N4.items()}

    def __getitem__(self, key):
        '''Make attributes accessible dict-like.'''
        return getattr(self, key)


def first_kind_discrete(orientations, order=4):
    '''
    Calc orientation tensors of first kind for given discrete vectors
    '''
    # Normalize orientations
    orientations = [np.array(v) / np.linalg.norm(v) for v in orientations]

    # Symmetrize orientations
#    orientations_reversed = [-v for v in orientations]
#    orientations = orientations + orientations_reversed

    einsumStrings = {
        1:  'ij             -> j',
        2:  'ij, ik         -> jk',
        3:  'ij, ik, il     -> jkl',
        4:  'ij, ik, il, im -> jklm',
        5:  'ij, ik, il, im, in     -> jklmn',
        6:  'ij, ik, il, im, in, ip -> jklmnp',
        }

    ori = orientations
    if order == 1:
        N = 1./len(orientations) * np.einsum(
                                        einsumStrings[order],
                                        ori,
                                        )
    elif order == 2:
        N = 1./len(orientations) * np.einsum(
                                        einsumStrings[order],
                                        ori, ori
                                        )
    elif order == 3:
        N = 1./len(orientations) * np.einsum(
                                        einsumStrings[order],
                                        ori, ori, ori
                                        )
    elif order == 4:
        N = 1./len(orientations) * np.einsum(
                                        einsumStrings[order],
                                        ori, ori, ori, ori
                                        )
    elif order == 5:
        N = 1./len(orientations) * np.einsum(
                                        einsumStrings[order],
                                        ori, ori, ori, ori, ori
                                        )
    elif order == 6:
        N = 1./len(orientations) * np.einsum(
                                        einsumStrings[order],
                                        ori, ori, ori, ori, ori, ori
                                        )
    else:
        raise Exception('Not implemented')

    return N
