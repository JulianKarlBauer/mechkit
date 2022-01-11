# # Notation: Converter Numeric

import mechkit
import numpy as np
import sympy as sp
import itertools

np.set_printoptions(
    linewidth=140,
    precision=3,
    # suppress=False,
)
converter = mechkit.notation.Converter()

# ### Mandel6: symmetric, Mandel9: Mandel6 + asymmetric parts
basics = mechkit.tensors.Basic()

# Symmetric identity
print(converter.to_mandel6(basics.I4s))
print(converter.to_mandel9(basics.I4s))

# Asymmetric identity
print(converter.to_mandel6(basics.I4a))
print(converter.to_mandel9(basics.I4a))

# ### Convenient autodetection of the notation: Specify only target

tensor = np.ones((3, 3, 3, 3))
mandel6 = converter.to_mandel6(tensor)
# Pass through
print(converter.to_tensor(mandel6))
print(converter.to_mandel6(mandel6))
print(converter.to_mandel9(mandel6))

# ### Vectorized explicit converter

expl_converter = mechkit.notation.ExplicitConverter()

tensors = np.random.rand(
    3, 3, 3, 3, 5, 2
)  # We have 5 times 2 tensors of fourth order

mandel6s = expl_converter.convert(
    inp=tensors, source="tensor", target="mandel6", quantity="stiffness"
)

for i in range(5):
    for j in range(2):
        print(f"Tensor at position {i}, {j} in Mandel6 notation")
        print(mandel6s[..., i, j])
