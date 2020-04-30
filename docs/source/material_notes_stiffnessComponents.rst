.. _DefinitionStiffnessComponents:

Definition of stiffness components
----------------------------------

Tensor components: (See [Betram2015]_ page 99 for details.)

.. math::
    \begin{align*}
        \mathbb{C}
        &=
        C_{ijkl}
        \;
        \mathbf{e}_{i}
        \otimes
        \mathbf{e}_{j}
        \otimes
        \mathbf{e}_{k}
        \otimes
        \mathbf{e}_{l}\\
    \end{align*}

Matrix components:

.. math::
    \begin{align*}
        \mathbb{C}
        &=
        \begin{bmatrix}
     C_{11}  & C_{12}       & C_{13} & C_{14} & C_{15} & C_{16} \\
             & C_{22}       & C_{23} & C_{24} & C_{25} & C_{26} \\
             &              & C_{33} & C_{34} & C_{35} & C_{36} \\
             &              &        & C_{44} & C_{45} & C_{46} \\
             & \text{sym}   &        &        & C_{55} & C_{56} \\
             &              &        &        &        & C_{66}
        \end{bmatrix}_{[\text{Voigt}]}      \hspace{-10mm}
        \scriptsize{
            \boldsymbol{V}_{\alpha} \otimes \boldsymbol{V}_{\beta}
            }   \\
        &=
        \begin{bmatrix}
     C_{11}  & C_{12}       & C_{13} & \sqrt{2}C_{14} & \sqrt{2}C_{15} & \sqrt{2}C_{16} \\
             & C_{22}       & C_{23} & \sqrt{2}C_{24} & \sqrt{2}C_{25} & \sqrt{2}C_{26} \\
             &              & C_{33} & \sqrt{2}C_{34} & \sqrt{2}C_{35} & \sqrt{2}C_{36} \\
             &              &        & 2C_{44} & 2C_{45} & 2C_{46} \\
             & \text{sym}   &        &         & 2C_{55} & 2C_{56} \\
             &              &        &         &         & 2C_{66}
        \end{bmatrix}_{[\text{Mandel6}]}    \hspace{-15mm}
        \scriptsize{
            \boldsymbol{B}_{\alpha} \otimes \boldsymbol{B}_{\beta}
            }
    \end{align*}

with

- :math:`\boldsymbol{B}_{\alpha}` : Base dyad of Mandel6 notation
  (See :mod:`mechkit.notation`)
- :math:`\boldsymbol{V}_{\alpha}` : Base dyad of Voigt notation
  (See [csmbrannonMandel]_)



