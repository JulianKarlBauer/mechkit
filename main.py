import mechkit
import numpy as np
import copy

np.set_printoptions(
    linewidth=140,
    precision=4,
    # suppress=False,
)

con = mechkit.notation.AbaqusConverter(silent=True)

mandel = np.array([1., 2, 3, 4, 5, 6])

voigt = con.mandel6_to_voigt(inp=mandel, voigt_type="stress")
print("Voigt")
print(voigt)

umat = con.mandel6_to_umat(inp=mandel, voigt_type="stress")
print("Umat")
print(umat)


