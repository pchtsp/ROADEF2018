import os
from distutils.core import setup
from Cython.Build import cythonize

files = ["geometry.pyx", "nodes_checks.pyx", "nodes.pyx", "nodes_optim.pyx"]
files_paths = [os.path.join("package", f) for f in files]

setup(name='Hello world app',
      ext_modules=cythonize(files_paths, language_level=3, annotate=True)
      )