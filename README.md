[![PyPI version](https://badge.fury.io/py/mechkit.svg)][url_pypi_mechkit]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3898702.svg)][url_latest_doi]
[![Documentation status](https://readthedocs.org/projects/mechkit/badge/?version=latest)][url_read_the_docs_latext]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JulianKarlBauer/mechkit/HEAD)
[![status](https://joss.theoj.org/papers/5bc306fb511a596eb25e2032a39b4baa/status.svg)](https://joss.theoj.org/papers/5bc306fb511a596eb25e2032a39b4baa)


<p align="center">
  <a href="https://github.com/JulianKarlBauer/mechkit/blob/67f1b61afbcac739a0bf61801a81d9a9028abb56/logo/logo.png">
  <img alt="mechkit" src="docs/source/images/logo/logo.png" width="50%">
  </a>
</p>

# mechkit

The Python package `mechkit` is a toolkit for researchers
in the field of continuum mechanics and material modeling.
`mechkit` contains methods and operators
for elementary tasks concerning tensor algebra and tensor notation,
focusing on second- and fourth-order tensors.

Please see [license][url_license],
[acknowledgment](#acknowledgment)
and cite the latest [Zenodo-DOI][url_latest_doi].

## Documentation

Documentation is given in [the docs][url_read_the_docs_latext].  

## Examples

Rendered example notebooks are in [the docs][url_read_the_docs_latest_notebooks].
These notebooks and corresponding scripts can be found [here][url_docs_source_notebooks].

## Installation [![PyPI version](https://badge.fury.io/py/mechkit.svg)][url_pypi_mechkit]

Install with `pip` following instructions on [Python Package Index][url_pypi_mechkit], i.e.,

```bash
pip install mechkit
```

**or** install from local files

- [Clone][url_how_to_clone] this repository to your machine
- Open a terminal and navigate to your local clone
- Install the package from the local clone into the current [env][url_env_python]i[ronment][url_env_conda] in develop mode:
	```shell
	python setup.py develop
	```

Note: [Develop vs. install](https://stackoverflow.com/a/19048754/8935243)

## Acknowledgment

The research documented in this repository has been funded by the 
[German Research Foundation (DFG, Deutsche Forschungsgemeinschaft)][dfg_website] - project number [255730231][dfg_project].
The support by the German Research Foundation within the International Research Training Group 
[“Integrated engineering of continuous-discontinuous long fiber reinforced polymer structures“ (GRK 2078)][grk_website]
is gratefully acknowledged.

[grk_website]: https://www.grk2078.kit.edu/
[dfg_website]: https://www.dfg.de/
[dfg_project]: https://gepris.dfg.de/gepris/projekt/255730231

[url_license]: LICENSE
[url_latest_doi]: https://doi.org/10.5281/zenodo.3898702
[url_how_to_clone]: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

[url_env_python]: https://docs.python.org/3/tutorial/venv.html
[url_env_conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

[url_read_the_docs_latext]: http://mechkit.readthedocs.io/?badge=latest
[url_read_the_docs_latest_notebooks]: https://mechkit.readthedocs.io/en/latest/source/example_notebooks.html
[url_docs_source_notebooks]: https://github.com/JulianKarlBauer/mechkit/tree/master/docs/source/notebooks
[url_pypi_mechkit]: https://pypi.org/project/mechkit/




