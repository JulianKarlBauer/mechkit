---
title: 'Mechkit: A continuum mechanics toolkit in Python'
tags:
  - Python
  - mechanics
  - continuum-mechanics
  - mechanics-of-materials
  - linear elasticity
  - fiber orientation tensors
  - notation
authors:
  - name: Julian Karl Bauer^[corresponding author]
    orcid: 0000-0002-4931-5869
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Philipp Lothar Kinon
    orcid: 0000-0002-4128-5124
    affiliation: "1"
  - name: Jonas Hund
    orcid: 0000-0003-2140-4285
    affiliation: 2
affiliations:
 - name: Institute of Mechanics, Karlsruhe Institute of Technology (KIT), Germany
   index: 1
 - name: Department of Structural Engineering, Norwegian University of Science and Technology (NTNU), Norway
   index: 2
# - name: Independent Researcher
#   index: 3
date: 02 February 2022
bibliography: paper.bib
---

# Summary

The Python package `mechkit` is a toolkit for researchers
in the field of continuum mechanics and material modeling.
`mechkit` contains methods and operators
for common tasks concerning tensor algebra and tensor notation.

In the context of engineering applications in three spatial dimensions, deformations and stresses in solids are
usually described by second-order tensors.
As linear mappings between observed deformations and possibly causal stresses,
fourth-order tensors have a special role in the field of linear elasticity.
To this end, the methods in `mechkit` are focussed on second- and fourth-order tensors. Main motivations can thus be found in the research concerning
linear elasticity
[@Bertram2015], [@Mandel1965], [@Fedorov1968], [@Mehrabadi1990], [@Thomson1856],
[@Cowin1992], [@Rychlewski2000], [@Spencer1970], [@Boehlke2001], [@Brannon2018]
and the description of microstructures of fiber-reinforced
composite materials
[@Bauer2021], [@Kanatani1984], [@Advani1987].

The implementations in `mechkit` aim at usability, seek to provide understandable source code,
and do not put major emphasis on performance.
Furthermore, the implementations follow, as directly as possible,
the notation and formulation in the respective scientific sources.
A redundant implementation of identical operations based on different
sources is aimed at, for validation reasons.

# Statement of need

The methods and operators for the description of linear elasticity and material modeling are mathematically simple and can be expressed in a compact manner.
Due to this relative simplicity, they are usually implemented independently by each scientist for use in their studies.
Consequentially, unlike in other research fields such as physics and computer science, there is no library for the common methods and operators of linear elasticity and modeling to this day.
This presents a major obstacle with regard to the exchange and reliability of research code as well as the interpretation and comparison of results.

The main goal of `mechkit` is reusable research code that increases the reliability of results and
accelerate as well as simplifies further research.
`mechkit` is inspired by [@fiberoripy] and the projects of the author of [@meshio].

## Motivation by example: Isotropic material and notations

In the overlapping area of theoretical continuum mechanics, experimental
material modeling,
and numerical solution methods for boundary value problems,
a multitude of different notations exist.
As an example, one may consider the description of the mechanical properties of a
homogeneous and isotropic, i.e. direction-independent, material within the framework of linear elasticity.
Such a material can be described identically by two scalar material parameters.
However, in the disciplines mentioned above, at least six different material parameters are commonly used,
motivated by different applications and measurement methods.
This results in fifteen possible combinations of scalar descriptions of an
isotropic material, which can be combined to the corresponding fourth-order elasticity tensor.
For this tensor, again, different notations exist. They either follow the
Voigt or Kelvin-Mandel notation or take account of the interfaces between open and commercial finite element codes.

The translation between different notations is often tedious and prone to errors.
`mechkit` allows an easy exchange between different notations with user-friendly
interfaces, thereby preventing errors.
The necessary number of translation functions between different notations
increases drastically with an increasing number of notations.
Consequentially, for even a small number of different notations, the implementation of all corresponding translation methods is not feasible.
Therefore, `mechkit` does not necessarily directly translate one notation into another.
Instead, in the case of the translation of second- and fourth-order tensors,
`mechkit` determines the shortest path between source and target in the set of implemented notations as illustrated in
the graph of currently supported notations, see \autoref{fig:stiffness_graph}.
This procedure greatly facilitates the addition of further notations to `mechkit`.
Essentially, only a translation function from and to a new notation has to be added to the existing code, to make translations from and to this new notation universally available in `mechkit`.

![Currently supported notations and translations of fourth-order stiffness tensors.\label{fig:stiffness_graph}](./figures/stiffness_graph.pdf){ width=60% }

# Acknowledgements

We acknowledge support from
<?insert here after contacted?>
during the genesis of this project.
We acknowledge contributions from Lisa Latussek.

The research documented in this manuscript has been funded by the German Research Foundation (DFG) within the International Research Training Group “Integrated engineering of continuous-discontinuous long fiber-reinforced polymer structures” (GRK 2078/2). The support by the German Research Foundation (DFG) is gratefully acknowledged.

# References
