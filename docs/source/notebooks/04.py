# # Deviator Operators: Application to Fiber Orientation Tensors

import mechkit
import numpy as np

np.set_printoptions(
    linewidth=140,
    precision=3,
    # suppress=False,
)
converter = mechkit.notation.Converter()

# Define convenient function
def print_N2_and_deviators(N4_mandel):
    N4 = converter.to_tensor(N4_mandel)
    dev_N4 = converter.to_mandel6(mechkit.operators.dev(N4, order=4))

    N2 = np.einsum("ijkk->ij", N4)
    dev_N2 = mechkit.operators.dev(N2, order=2)

    print("N4=")
    print(N4_mandel)
    print("dev_N4=")
    print(dev_N4)

    print("N2=")
    print(N2)
    print("dev_N2=")
    print(dev_N2)


# ## Isotropic N4: No deviation from the isotropic fourth order fiber orientation tensor
print("Isotropic")
print_N2_and_deviators(N4_mandel=mechkit.fabric_tensors.Basic().N4["iso"])

# ## Unidirectional N4: Large deviations from isotropic fourth order fiber orientation tensor
print("Unidirectional")
print_N2_and_deviators(N4_mandel=mechkit.fabric_tensors.Basic().N4["ud_x"])
