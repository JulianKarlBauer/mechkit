import setuptools
from os import path
import sys

if sys.version_info > (3, 0):
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Basic continuum mechanics toolkit"

setuptools.setup(
    name="mechkit",
    version="0.3.0",
    author="Julian Karl Bauer",
    author_email="JulianKarlBauer@gmx.de",
    description="Basic continuum mechanics toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JulianKarlBauer/mechkit",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "networkx; python_version > '2.7'",
        "decorator == 4.3.0; python_version <= '2.7'",
        "networkx == 2.2; python_version <= '2.7'",
    ],
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
