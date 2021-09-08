#!/usr/bin/env python

from distutils.core import setup

setup(
    name="bayeshist",
    version="0.1",
    description="Bayesian histograms for estimation of binary event rates",
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
