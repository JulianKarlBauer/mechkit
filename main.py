#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Main script developing mechkit
'''

import numpy as np
import importlib
import mechkit as mk
importlib.reload(mk)

np.set_printoptions(
        linewidth=140,
        precision=2,
        # suppress=False,
        )

import mechkit as mk
con = mk.notation.Converter()
tensors = mk.tensors.basic()

# Define what to print
printQueue = [
        "'Hi'",
        ]

for val in printQueue:
    print(val)
    print(eval(val), '\n')
