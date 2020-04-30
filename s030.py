import mechkit
import numpy as np

np.set_printoptions(
    linewidth=140,
    precision=4,
    # suppress=False,
)

kwargs = {"E_l": 100.0, "E_t": 20.0, "nu_lt": 0.3, "G_lt": 10.0, "G_tt": 7.0}

m = mechkit.material.TransversalIsotropic(**kwargs)
m2 = mechkit.material.TransversalIsotropic(**kwargs, principal_axis=[0, 1, 0])
m3 = mechkit.material.TransversalIsotropic(**kwargs, principal_axis=[0, 0, 1])
