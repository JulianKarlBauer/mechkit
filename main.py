#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main script developing mechkit
"""

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
tensors = mk.tensors.Basic()

t2 = np.array([[1.0, 6.0, 5.0,], [6.0, 2.0, 4.0,], [5.0, 4.0, 3.0,],])

# Define what to print
printQueue = [
    "con.to_mandel6(t2)",
    "np.sqrt(2.)",
]

for val in printQueue:
    print(val)
    print(eval(val), "\n")
