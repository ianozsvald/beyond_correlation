#!/usr/bin/env python
"""discover_feature_relationships tries to find column-to-column relationships in your dataframe using machine learning
"""

doclines = __doc__.split("\n")

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
    version="1.1",
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

