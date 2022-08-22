notation
--------

+-------------------------------------+-----------------------------------------------------------------+----------------------------------------+-------------+
| Converter                           | Supported notations                                             | Supported Quantities                   | Vectorized  |
+=====================================+=================================================================+========================================+=============+
| mechkit.notation.Converter          | tensor, mandel6, mandel9                                        | stress, strain, stiffness, compliance  | no          |
+-------------------------------------+-----------------------------------------------------------------+----------------------------------------+-------------+
| mechkit.notation.ExplicitConverter  | tensor, mandel6, mandel9, voigt, umat, vumat, (abaqusMatAniso)  | stress, strain, stiffness, compliance  | yes         |
+-------------------------------------+-----------------------------------------------------------------+----------------------------------------+-------------+

--------------------------------------------------------

.. autoclass:: mechkit.notation.Converter
    :members:
    :undoc-members:
    :noindex:
    :exclude-members: to_mandel6, to_mandel9, to_like, to_tensor

    .. :show-inheritance:

.. .. autoclass:: mechkit.notation.VoigtConverter
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:
..     :exclude-members: mandel6_to_voigt, voigt_to_mandel6


.. autoclass:: mechkit.notation.ExplicitConverter
    :members:
    :undoc-members:
    :noindex:
    :exclude-members: convert



.. autofunction:: mechkit.notation.get_mandel_base_sym

.. autofunction:: mechkit.notation.get_mandel_base_skw





