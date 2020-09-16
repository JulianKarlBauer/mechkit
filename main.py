import mechkit
import numpy as np
from mechkit.notation import TensorComponents

np.set_printoptions(
    linewidth=140,
    precision=4,
    # suppress=False,
)

con = mechkit.notation.AbaqusConverter(silent=True)

a = TensorComponents(np.arange(9).reshape(3, 3))

tensor = TensorComponents(
    np.arange(9).reshape(3, 3), quantity="stress", notation="tensor"
)
mandel6 = TensorComponents(np.arange(6), quantity="stress", notation="mandel6")
mandel9 = TensorComponents(np.arange(9), quantity="stress", notation="mandel9")


# mandel = np.array([1., 2, 3, 4, 5, 6])
#
# voigt = con.mandel6_to_voigt(inp=mandel, voigt_type="stress")
# print("Voigt")
# print(voigt)
#
# umat = con.mandel6_to_umat(inp=mandel, voigt_type="stress")
# print("Umat")
# print(umat)
