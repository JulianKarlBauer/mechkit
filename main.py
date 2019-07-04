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
I_mandel6 = con.to_mandel(mk.tensors.I4s)
