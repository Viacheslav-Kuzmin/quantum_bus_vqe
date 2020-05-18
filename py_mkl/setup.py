from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

ext_modules = cythonize([Extension("py_mkl", ["py_mkl.pyx"], language='c')])


setup(
  name = 'Hello world app',
  cmdclass = {'build_ext': build_ext},
  include_dirs=[np.get_include()],
  ext_modules = ext_modules
)

# python setup.py build_ext --inplace
