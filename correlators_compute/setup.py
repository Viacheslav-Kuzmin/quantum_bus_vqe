from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

ext_modules = cythonize(
    [
        Extension("corr_ssh_comp", ["corr_ssh_comp.cc"], 
        language='c++', 
#         extra_compile_args=['-std=c++14'],
        library_dirs=['/home/slava/itensor/lib_a', '/usr/lib'],
        libraries = ['itensor', 'pthread', 'blas', 'lapack'],
                  ),
    ]
)

setup(
  name = 'corr_ssh_comp',
  cmdclass = {'build_ext': build_ext},
  include_dirs=['/home/slava/itensor', '/home/slava/itensor/itensor', '/home/slava/miniconda3/include'],
  ext_modules = ext_modules
)

# python setup.py build_ext --inplace