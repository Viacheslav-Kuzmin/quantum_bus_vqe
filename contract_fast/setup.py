from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

ext_modules = cythonize(
    [
        Extension("contract_fast", ["contract_fast.pyx"], 
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        language='c')
    ]
)


setup(
  name = 'Hello world app',
  cmdclass = {'build_ext': build_ext},
  include_dirs=[np.get_include()],
  ext_modules = ext_modules
)

# python setup.py build_ext --inplace

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize

# ext_modules = [
#     Extension(
#         "hello",
#         ["hello.pyx"],
#         extra_compile_args=['-fopenmp'],
#         extra_link_args=['-fopenmp'],
#     )
# ]

# setup(
#     name='hello-parallel-world',
#     ext_modules=cythonize(ext_modules),
# )