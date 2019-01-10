#!/usr/bin/env python
"""discover_feature_relationships tries to find column-to-column relationships in your dataframe using machine learning
"""
import io
import re

doclines = __doc__.split("\n")

# stolen from: https://github.com/pallets/flask/blob/master/setup.py
with io.open('discover_feature_relationships/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)


# Chosen from http://www.python.org/pypi?:action=list_classifiers
classifiers = """\
Development Status :: 5 - Production/Stable
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: Free To Use But Restricted
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python
Topic :: Software Development :: Libraries :: Python Modules
Topic :: Software Development :: Testing
"""

from setuptools import setup, find_packages
setup(
    name="discover_feature_relationships",
    version=version,
    license='MIT',
    url="https://github.com/ianozsvald/discover_feature_relationships",
    author="Ian Ozsvald",
    author_email="ian@ianozsvald.com",
    maintainer="Ian Ozsvald",
    maintainer_email="ian@ianozsvald.com",
    description=doclines[0],
    long_description = """TODO
    """,
    long_description_content_type='text/markdown',
    classifiers=filter(None, classifiers.split("\n")),
    platforms=["Any."],
    packages=find_packages(),
    #install_requires=['']
)

