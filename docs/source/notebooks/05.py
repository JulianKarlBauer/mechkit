# # Operators: Sym

import mechkit
import numpy as np

np.set_printoptions(
    linewidth=140,
    precision=2,
    suppress=True,
)
np.random.seed(1)
converter = mechkit.notation.Converter()


def print_in_mandel9(tensor):
    print(converter.to_mandel9(tensor))


# ## Split random tensor of fourth order into completely symmetric and skew parts
tensor = np.random.rand(3, 3, 3, 3)
sym_part = mechkit.operators.sym(
    tensor, sym_axes=None
)  # by default all axes are symmetrized
skew_part = tensor - sym_part

print("tensor=")
print_in_mandel9(tensor)

print("sym_part=")
print_in_mandel9(sym_part)

print("skew_part=")
print_in_mandel9(skew_part)

print("sym_part + skew_part")
print_in_mandel9(sym_part + skew_part)
assert np.allclose(tensor, sym_part + skew_part)

# ## Split into part which has inner symmetry and the remaining part
sym_inner_part = mechkit.operators.Sym_Fourth_Order_Special(label="inner")(tensor)
remaining = tensor - sym_inner_part

print("tensor=")
print_in_mandel9(tensor)

print("sym_inner_part=")
print_in_mandel9(sym_inner_part)

print("remaining=")
print_in_mandel9(remaining)
