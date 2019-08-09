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
tensors = mk.tensors.Basic()

t2 = np.array(
    [[1., 6., 5., ],
     [6., 2., 4., ],
     [5., 4., 3., ], ]
    )

# Define what to print
printQueue = [
        "con.to_mandel6(t2)",
        'np.sqrt(2.)',
        ]

for val in printQueue:
    print(val)
    print(eval(val), '\n')
