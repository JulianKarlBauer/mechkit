import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="mechkit",
    version="0.0.2",
    author="Julian Karl Bauer",
    author_email="JulianBauerKIT@gmail.com",
    description="Basic tools for continuum mechanics using numpy",
    long_description="Basic tools for continuum mechanics using numpy",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/JulianBauerKIT/mechkit",
    packages=setuptools.find_packages(),
    install_requires=[
          'numpy',
          ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
