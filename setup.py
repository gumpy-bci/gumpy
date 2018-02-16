#!/usr/bin/env python

# Copyright (c) 2017-2018 The gumpy developers:
#
# 2017-2018  Zied Tayeb         <zied.tayeb@tum.de>
# 2017-2018  Nicolai Waniek     <rochus+gumpy@rochus.net>
# 2017-2018  Juri Fedjaev
# 2017-2018  Leonard Rychly
#

"""EEG signal processing and classification toolbox.

This toolbox provides signal processing functions and classes to work with BCI
datasets. Many of the functions internally call existing libraries for signal
processing or numerical computation such as ``numpy`` or ``scipy``. In these
cases the functions are called with parameters that were found to be suitable
for BCI computing and brain machine interfaces.

The name of the toolbox is a reference to the Gumby Brain Specialist sketch by
Monty Python.
"""

DISTNAME         = 'gumpy'
DESCRIPTION      = 'EEG signal processing and classification toolbox'
LONG_DESCRIPTION = __doc__
MAINTAINER       = 'The gumpy developers'
MAINTAINER_MAIL  = 'zied.tayeb@tum.de'
LICENSE          = 'MIT'
URL              = 'www.gumpy.org'

# extract version from source file
VERSION_DATA = {}
with open('gumpy/version.py') as fp:
    exec(fp.read(), VERSION_DATA)
    VERSION = VERSION_DATA['__version__']


from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Human Machine Interfaces',
          ],
          install_requires = [
            'numpy',
            'scipy',
            'scikit-learn',
            'seaborn',
            'pandas',
            'PyWavelets',
            'mlxtend',
          ],
          name=DISTNAME,
          version=VERSION,
          description=DESCRIPTION,
          long_description=__doc__,
          url=URL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_MAIL,
          license=LICENSE,
          packages=find_packages(exclude=['tests*']),
          python_requires='>=3',
          zip_safe=False)
