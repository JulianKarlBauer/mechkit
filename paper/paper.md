---
title: 'Mechkit: A continuum mechanics toolkit in Python'
tags:
  - Python
  - mechanics
  - continuum-mechanics
  - mechanics-of-materials
  - tensor calculus
  - notation
authors:
  - name: Julian Karl Bauer^[co-first author]
    orcid: 0000-0002-4931-5869
    affiliation: "1" # (Multiple affiliations must be quoted)
#  - name: Author Without ORCID^[co-first author] # note this makes a footnote saying 'co-first author'
#    affiliation: 2
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

Das Python package `mechkit` ist eine Werkzeugkiste für Forscher
im Bereich Continuums Mechanik und Materialmodellierung.
`Mechkit` beinhaltet Methoden und Operatoren,
die häufig auftretende Aufgaben, insbesondere im Bereich Tensoralgebra und Notation,
vereinfachen.

Deformationen und Spannungen in Festkörpern unserer dreidimensionalen Welt werden
im Rahmen ingeniersmäßiger Anwendungen üblicherweise durch Tensoren zweiter Stufe beschrieben.

Als lineare Abbildungen zwischen den beobachteten Deformationen und ggf. ursächlichen Spannungen,
wird Tensoren vierter Stufe eine besondere Aufgabe im Bereich der linearen Elastizität zuteil.
Eine zentrale Implementierung der
Tensoren zweiter und vierter Stufe sind das primäre Anwendungsfeld für die Methoden in `mechkit`
und sind motiviert durch Forschung in den Bereichen
linearer Elastizität und der Beschreibung von Mikrostrukturen faserverstärkter
Compositwerkstoffe.


Die Implementierungen zielen auf Einfachheit in der Anwendung sowie verständlichen Quellcode ab
und legen keinen primären Wert auf Performance.
Desweiteren folgen die Implementierungen möglichst direkt der Notation und Formulierung
der Formeln in den wissenschaftlichen Quellen.
Eine redundante Implementierung identischer Operationen basierend auf verschiedenen
Quellen wird angestrebt.



# Statement of need

Die im Bereich der linearen Elastizität und Materialmodellierung auftretenden Methoden und Operatoren sind
vergleichsweise einfach und kompakt.
Aufgrund dieser Einfachheit werden sie üblicherweise bereits im Studium oder zu Beginn
einer weiterführenden wissenschaftlchen Ausbildung von jedem Wissenschaftler separat
implementiert und finden keinen Eingang in allgemeinere Bibliotheken z.B. aus dem Bereich der Physik.
Abgesehen von didaktischen Vorteilen, verursacht dieses Vorgehen Probleme
beim Austausch von Forschungscode und
der Zuverlässigkeit der Implementierungen.

Das Hauptziel des Projektes `mechkit` ist die Wiederverwendung von Forschungscode,
um die Zuverlässigkeit der Forschungsergebnisse zu erhöhen und
die weitere Forschung zu beschleunigen und zu vereinfachen.

Im Überlappungsbereich der theoretischen Continuums Mechanik, der experimenteleln
Materialmodellierung
und der numerischen Lösungsmethoden für Randwertprobleme
besteht eine Vielzahl unterschiedlicher Notationen.

### Beispiel: Notationen Isotropes Material

Als Beispiel sei die Beschreibung der mechanischen Eigenschaften eines
homognene und isotropen, d.h. richtungsunabhängigen, Materials im Rahmen der
einfachsten Theorie, der linearen Elastizität, skizziert.
Ein solches Material kann durch zwei skalare Materialparameter identisch beschrieben werden.
In den oben genannten Disziplinen sind jedoch mindestens sechs verschiedene Materialparameter üblich
und durch verschiedene Anwendungen und Messverfahren motiviert.
Dadurch ergeben sich 15 mögliche Kombinationen von skalaren Beschreibungen eines
isotropen Materials, welche zu einem Tensor vierter Stufe kombiniert werden können.
Für den resultierenden Tensor existieren wiederum verschiedene Notationen nach
Voigt, Kelvin-Mandel und offene und kommerzielle Finite-Elemente-Codes.

`Mechkit` ermöglicht eine einfache Überführung zwischen den genanten Notationen
mit anwendungsfreundlichen Schnittstellen und hilft dadurch Fehler zu vermeiden.

`Mechkit` is inspiriert von [@fiberoripy] sowie den Projekten des Autors von [@meshio]
und findet neben zahlreichen closed-source Projekten, Verwendung in
[@mechmean].


<!-- `Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike. -->

# Structure

<!-- # Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text. -->

<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png){ width=20% }
and referenced from text using \autoref{fig:example}. -->


# Acknowledgements

We acknowledge support from

during the genesis of this project.
We acknowledge contributions from Lisa Latussek.

The research documented in this manuscript has been funded by the German Research Foundation (DFG) within the International Research Training Group “Integrated engineering of continuous-discontinuous long fiber-reinforced polymer structures” (GRK 2078/2). The support by the German Research Foundation (DFG) is gratefully acknowledged.

# References