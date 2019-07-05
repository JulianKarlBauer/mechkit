#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Main script developing mechkit
'''

import numpy as np
import mechkit as mk
import importlib
importlib.reload(mk)

np.set_printoptions(
        linewidth=140,
        precision=2,
        # suppress=False,
        )

con = mk.notation.Converter()
tensors = mk.tensors.basic()


# Define what to print
printQueue = [
        'tensors.I2',
        'con.to_mandel6(tensors.I2)',
        'np.arange(9).reshape(3,3)',
        'con.to_mandel6(np.arange(9).reshape(3,3))',
        'tensors.I4s',
        'con.to_mandel6(tensors.I4s)',
        'con.to_mandel9(tensors.I4s)',
        'con.to_mandel9(tensors.I4a)',
        ]


for val in printQueue:
    print(val)
    print(eval(val), '\n')
