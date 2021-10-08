#!/usr/bin/env python

from distutils.core import setup

from codecs import open
import os

from bayeshist import __version__

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bayeshist",
    license="MIT",
    version=__version__,
    description="Bayesian histograms for estimation of binary event rates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dion Häfner",
    author_email="mail@dionhaefner.de",
    url="https://github.com/dionhaefner/bayeshist",
    packages=["bayeshist"],
    install_requires=[
        "numpy",
        "scipy",
    ],
    python_requires=">=3.6",
)
