[docs]

# Mechkit

Basic tools for continuum mechanics using numpy developed at KIT

## Outline:
- Done:
    - Notation / Converter
    - Tensors


- Todo:
    - Meanfield
        - Localizations
        - Approximations
    - Fabric tensors
        - Discrete fibers
        - Fiber orientation distribution
        - Visualization
        - Orientation averaging in meanfield
    - Fatigue (Miners rule with Woehler curves)
        - Identify time signal of scalar stress from simulation
          - Superposition in linear analyses
          - Surface points
          - Scalar stress measures
          - Critical plane
        - Calc [Markov matrix][markov] by rain flow counting for given
          - Time signal of stress
          - Discretization
        - Calc collective for given
          - Markov matrix
          - Influence of mean stresses (Haigh diagram) (Scale amplitude)
        - Calc damage for given
          - Collective
          - Woehler curve
          - Miner rule
            - (Original, Elementary, Haibach, Liu-Zenner, Consistent)

[markov]: http://archiv.windenergietage.de/WT25/25WT1011_F5_0935_EUROS.pdf

[docs]: https://readthedocs.org/projects/pip/badge/

--------------------------------------------------------------------------
# Development
- Ubuntu 16.04
- Anaconda 4.6.14

## Environment
    conda create -n mechkit python=3
    source activate mechkit
    conda install numpy
    conda install ipython

## Documentation
    conda install sphinx
    sphinx-quickstart

Add extensions:
- mathjax
- viewcode
- autodoc
- napoleon

Extend path to enable autodoc to find source code

    sys.path.insert(0, os.path.abspath(os.path.join('..',)))

Change theme

    html_theme = 'sphinx_rtd_theme'

Create doc

    make clean
    make html

## Packaging for PyPi
Follow [PyPa](https://packaging.python.org/tutorials/packaging-projects/)

- Add setup.py
- Add LICENSE

### Build
- Install
        conda install setuptools wheel
- Do
        python3 setup.py sdist bdist_wheel

### Upload
- Install
        conda install twine
- Do
        python3 -m twine upload dist/*

### Install from PyPi and check
Create virtual environment

    pip3 install --user virtualenv
    virtualenv v123
    source ./v123/bin/activate

Install

    python3 -m pip install mechkit

Deactivate virtualenv

    deactivate

## ReadTheDocs

- Log in to [ReadTheDocs](https://readthedocs.org) using github account.
- Add repository and click "Next"...

## Packaging for Conda
