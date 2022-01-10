# # Title

import mechkit
import numpy as np

np.set_printoptions(
    linewidth=140,
    precision=3,
    # suppress=False,
)

mat = mechkit.material.Isotropic(E=2e6, nu=0.3)
mat = mechkit.material.Isotropic(E=2e6, K=1e6)
mat1 = mechkit.material.Isotropic(M=15, G=5)
mat2 = mechkit.material.Isotropic(C11_voigt=20, C44_voigt=5)

printQueue = [
    "mat.G",
    "mat['E']",
    "mat1['stiffness_voigt']",
    "mat2['stiffness_voigt']",
    "(0.5*mat1 + 0.5*mat2)['stiffness_voigt']",
    "mat1['stiffness_mandel6']",
    "mat1['compliance_mandel6']",
]
for val in printQueue:
    print(val)
    print(eval(val), "\n")

mat = mechkit.material.TransversalIsotropic(
    E_l=100.0, E_t=20.0, nu_lt=0.3, G_lt=10.0, G_tt=7.0, principal_axis=[0, 1, 0]
)

printQueue = [
    "mat.compliance_voigt",
    "mat.stiffness_mandel6",
]
for val in printQueue:
    print(val)
    print(eval(val), "\n")
