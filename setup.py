import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="mechkit",
    version="0.2.6",
    author="Julian Karl Bauer",
    author_email="JulianKarlBauer@gmx.de",
    description="Basic continuum mechanics toolkit",
    long_description="Basic continuum mechanics toolkit",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/JulianKarlBauer/mechkit",
    packages=setuptools.find_packages(),
    install_requires=[
          'numpy',
          ],
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
