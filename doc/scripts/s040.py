import numpy as np
from numpy.linalg import norm

np.set_printoptions(
    linewidth=140,
    precision=4,
    # suppress=False,
)


if True:
    a = np.random.rand(3, 1)
    a = a / norm(a)

    b = np.random.rand(3, 1)
    b = b / norm(b)
else:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 0.0])


# Reshape to get Matlab-like operations and normalize
a = a.reshape(3, 1) / norm(a)
b = b.reshape(3, 1) / norm(b)

c = a + b
R = 2.0 * (c @ c.T) / (c.T @ c) - np.eye(3)
# Thanks to https://math.stackexchange.com/a/2672702/694025

assert np.allclose(R @ a, b)
assert np.allclose(np.linalg.det(R), 1.)


# First try
# from numpy import cross
# c = cross(a, b)
# s = np.dot(a, b)
# R = np.eye(3) + np.outer(c, c) + ((1.-s) / norm(c)**2) * np.outer(c, c)
