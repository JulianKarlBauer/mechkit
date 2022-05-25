import mechkit
import numpy as np
from mechkit.notation import Components

np.set_printoptions(
    linewidth=140,
    precision=4,
    # suppress=False,
)

stress_tensor = Components(
    np.arange(9, dtype=np.float64).reshape(3, 3), quantity="stress", notation="tensor"
)
# assert np.allclose(stress_tensor, stress_tensor.to_mandel9().to_tensor())

strain_tensor = Components(
    np.arange(9, dtype=np.float64).reshape(3, 3), quantity="strain", notation="tensor"
)

r = strain_tensor.to_mandel6()

assert np.allclose(stress_tensor, stress_tensor.to_mandel9().to_tensor())

comp_tensor_bunch = Components(
    np.arange(324, dtype=np.float64).reshape(3, 3, 3, 3, 4),
    quantity="compliance",
    notation="tensor",
)

stress_voigt_bunch = Components(
    np.arange(1, 1 + 2 * 18, step=2, dtype=np.float64).reshape(6, 3),
    quantity="stress",
    notation="voigt",
)

# Vectorized
stiff_tensor_bunch = Components(
    np.arange(2 * 324, dtype=np.float64).reshape(3, 3, 3, 3, 2, 4),
    quantity="stiffness",
    notation="tensor",
)


stiff_mandel6_bunch = stiff_tensor_bunch.to_mandel6()
stiff_voigt_bunch = stiff_tensor_bunch.to_voigt()

stress_mandel6 = stress_tensor.to_mandel6()
stress_mandel9 = stress_tensor.to_mandel9()

stress_mandel6_bunch = stress_voigt_bunch.to_mandel6()
stress_mandel9_bunch = stress_voigt_bunch.to_mandel9()

# Check for order consistency
stiff_tensor_bunch.to_vumat().to_voigt()[0]
stiff_tensor_bunch.to_voigt()[0]

############
# Add way back
# s = stiff_mandel6_bunch.to_abaqusMatAniso()
# s.to_mandel6()


# mandel = np.array([1., 2, 3, 4, 5, 6])
#
# voigt = con.mandel6_to_voigt(inp=mandel, voigt_type="stress")
# print("Voigt")
# print(voigt)
#
# umat = con.mandel6_to_umat(inp=mandel, voigt_type="stress")
# print("Umat")
# print(umat)
