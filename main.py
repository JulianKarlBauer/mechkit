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
I_mandel6 = con.to_mandel6(mk.tensors.I4s)
I_mandel9 = con.to_mandel9(mk.tensors.I4s)


# Define what to print
printQueue = [
        'con.to_mandel6(I_mandel9)',
        'con.to_mandel9(I_mandel6)',
        ]


for val in printQueue:
    print(val)
    print(eval(val), '\n')
