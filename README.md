[![Documentation Status](https://readthedocs.org/projects/mechkit/badge/?version=latest)](http://mechkit.readthedocs.io/?badge=latest)
[![Anaconda-Server Badge](https://anaconda.org/anaconda/markdown/badges/installer/conda.svg)][conda]
[![PyPI version](https://badge.fury.io/py/mechkit.svg)][PyPi]

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
    - ...

[conda]: https://anaconda.org/JulianBauerKIT/mechkit
[PyPi]: https://pypi.org/project/mechkit/
[markov]: http://archiv.windenergietage.de/WT25/25WT1011_F5_0935_EUROS.pdf

--------------------------------------------------------------------------

# Ship new release
(See following sections for more details on single steps)

- Doc
        cd mechkit/docs
        make clean
        make html

- PyPi

    Change version-variable in `setup.py`

        source activate mechkit
        cd mechkit
        python3 setup.py sdist bdist_wheel
        python3 -m twine upload dist/*

- Conda
        cd mechkit/conda_build
        conda skeleton pypi mechkit
        ./conda_build_versions_environments.sh

----------------------------------------------------------------------
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
Follow [PyPa][PyPa]

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
- Create virtual environment

        pip3 install --user virtualenv
        virtualenv v123
        source ./v123/bin/activate

- Install

        python3 -m pip install mechkit

- Deactivate virtualenv

        deactivate

## ReadTheDocs

- Log in to [ReadTheDocs](https://readthedocs.org) using github account.
- Add repository and click "Next"...(https://packaging.python.org/tutorials/packaging-projects/)

## Packaging for Conda
Follow [Conda skeleton][conda_skeleton]

### Automated
Use [conda build script][conda_build_script_url] named `conda_build_versions_environments.sh`
to build the module
- for multiple python versions
- on multiple environments

    - Install

            conda install conda-build
            conda install anaconda-client

    - Create recipe (meta.yaml)

            mkdir conda_build
            cd conda_build
            conda skeleton pypi mechkit

    - Place `conda_build_versions_environments.sh` in directory `conda_build`

    - Specify the desired versions and environments by changing the script

    - Make script executable

            sudo chmod +x conda_build_versions_environments.sh

    - Execute

            ./conda_build_versions_environments.sh

    Note: Login may be required (`anaconda login`)

----------------------------------------------------------------------
### Manually (Just for completeness. Use automated version)

#### Build

- Install

        conda install conda-build

- Create recipe (meta.yaml)

        mkdir conda_build
        cd conda_build
        conda skeleton pypi mechkit

- Build

        conda-build mechkit
        conda-build --python 2.7 mechkit

- Install and test locally

        conda install --use-local mechkit

#### Convert to other environments
[conda_build_script]:

    mkdir tmp
    cd tmp
    conda convert -f --platform all ~/miniconda3/envs/mechkit/conda-bld/linux-64/mechkit-0.0.1-py27h39e3cac_0.tar.bz2

#### Upload to anaconda.org
- Install

        conda install anaconda-client

- Login

        anaconda login

- Upload

        anaconda upload ~/miniconda3/envs/mechkit/conda-bld/linux-64/mechkit-0.0.1-py37h39e3cac_0.tar.bz2

- Enable automatic upload of successful build:

        conda config --set anaconda_upload yes

- Install and test

        conda create -n test123 python=3
        conda install -c julianbauerkit mechkit

- Clean up testing

        conda deactivate
        conda info --envs
        conda remove -n test123 --all


### End manually
----------------------------------------------------------------------

[PyPa]: https://packaging.python.org/tutorials/packaging-projects/

[conda_skeleton]: https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html

[conda_build_script_url]: https://github.com/mcocdawc/chemcoord/blob/ae781f3360691cc1d4a654d5cb4f9dc0694dd7d3/conda.recipe/build_conda_packages.sh

