# # Materials

import mechkit
import numpy as np

np.set_printoptions(
    linewidth=140,
    precision=3,
    # suppress=False,
)

# Inspect some keyword combinations for constructor
mat = mechkit.material.Isotropic(E=2e6, nu=0.3)
mat1 = mechkit.material.Isotropic(E=2e6, K=1e6)
mat2 = mechkit.material.Isotropic(C11_voigt=20, C44_voigt=5)

# Address attributes directly or as dictionary
print(mat.G)
print(mat["E"])

# Get stiffness in common notations
print(mat["stiffness_voigt"])
print(mat["stiffness_mandel6"])

# Do arithmetic on material instances
print((0.5 * mat1 + 0.5 * mat2)["stiffness_voigt"])

# Get a transversally isotropic material
transv = mechkit.material.TransversalIsotropic(
    E_l=100.0, E_t=20.0, nu_lt=0.3, G_lt=10.0, G_tt=7.0, principal_axis=[0, 1, 0]
)
print(transv.stiffness_mandel6)
