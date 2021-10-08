#!/usr/bin/env python

from distutils.core import setup

from codecs import open
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# read version from __init__.py
version_file = os.path.join(here, "bayeshist", "__init__.py")
version_pattern = re.compile(r"__version__\s*=\s*[\"'](.+)[\"']")

with open(version_file, encoding="utf-8") as f:
    for line in f:
        match = version_pattern.match(line)
        if match:
            version = match.group(1)
            break
    else:
        raise RuntimeError("Could not determine version from __init__.py")


setup(
    name="bayeshist",
    license="MIT",
    version=version,
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
    zip_safe=False,
)
