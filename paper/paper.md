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
  - name: Philipp Lothar Kinon^[co-first author] # note this makes a footnote saying 'co-first author', feel free to hand over this item to another author
    orcid: 0000-0002-4128-5124
    affiliation: "1"
#  - name: Author with no affiliation^[corresponding author]
#    affiliation: 3
affiliations:
 - name: Institute of Mechanics, Karlsruhe Institute of Technology (KIT), Germany
   index: 1
# - name: Institution Name
#   index: 2
# - name: Independent Researcher
#   index: 3
date: 02 February 2022
bibliography: paper.bib
---

# Summary

The Python package `mechkit` is a toolkit for researchers
in the field of continuum mechanics and material modeling.
`Mechkit` contains methods and operators
that simplify common tasks - especially in the area of tensor algebra and tensor notation.

In the context of engineering applications within three spatial dimensions, deformations and stresses in solids are
usually described by second-order tensors.
As linear mappings between observed deformations and possibly causal stresses,
fourth-order tensors have a special task in the field of linear elasticity.
To this end, the methods in `mechkit` are primarily focussed on second- and fourth-order tensors. Main motivations can thus be found in the research in
linear elasticity
[@Bertram2015], [@Mandel1965], [@Fedorov1968], [@Mehrabadi1990], [@Thomson1856],
[@Cowin1992], [@Rychlewski2000], [@Spencer1970], [@Boehlke2001], [@Brannon2018]
and the description of microstructures of fiber reinforced
composite materials
[@Bauer2021], [@Kanatani1984], [@Advani1987].

The implementations in `mechkit` aim at usability, seek to provide understandable source code
and do not put major emphasis on performance.
Furthermore, the implementations follow, as directly as possible,
the notation and formulation in the respective scientific sources.
A redundant implementation of identical operations based on different
sources is strived for.


# Statement of need

The methods and operators occurring in the field of linear elasticity and material modeling are
comparatively simple and compact.
Due to this simplicity, they are usually implemented independently by each scientist
during the studies and
do not find their way into more general libraries which are common in other research fields such as physics or computer science.
Apart from didactic advantages, this procedure causes problems for
the exchange of research code and
the reliability of the implementations.

The main goal of the project `mechkit` is the reuse of research code,
to increase the reliability of the research results and
accelerate and simplify further research.
`Mechkit` is inspired by [@fiberoripy] and the projects of the author of [@meshio].

## Motivation by example: Isotropic material and notations

In the overlapping area of theoretical continuum mechanics, experimental
material modeling
and the numerical solution methods for boundary value problems
a multitude of different notations exist.
As an example, consider the description of the mechanical properties of a
homogeneous and isotropic, i.e. direction-independent, material within the framework of linear elasticity.
Such a material can be described identically by two scalar material parameters.
However, in the disciplines mentioned above, at least six different material parameters are commonly used
and motivated by different applications and measurement methods.
This results in fifteen possible combinations of scalar descriptions of an
isotropic material, which can be combined to a fourth-order elasticity tensor.
For this tensor, again, different notations exist. They either follow the
Voigt or Kelvin-Mandel notation or take account of the interfaces between open and commercial finite element codes.

`Mechkit` allows an easy exchange between the abovementioned notations with user-friendly
interfaces and thus helps to avoid errors.
Since the necessary number of translation functions between different notations
increases drastically with increasing number of notations,
translation between all notations might not be practical.
For the case of notations of second- and fourth-order tensors,
the shortest path between source and target notation is determined by
the graph of supported notations, see \autoref{fig:stiffness_graph}.

![Currently supported notations of fourth-order stiffness tensors.\label{fig:stiffness_graph}](stiffness_graph.png){ width=60% }


# Acknowledgements

We acknowledge support from
<?insert here after contacted?>
during the genesis of this project.
We acknowledge contributions from Lisa Latussek.

The research documented in this manuscript has been funded by the German Research Foundation (DFG) within the International Research Training Group “Integrated engineering of continuous-discontinuous long fiber-reinforced polymer structures” (GRK 2078/2). The support by the German Research Foundation (DFG) is gratefully acknowledged.

# References
