from distutils.core import setup, Extension
import numpy

setup(name='dga',
      version='1.0',
      description='Dynamical vertex correction solver',
      author='Paul Worm',
      author_email='pworm(at)gmail.com',
      url='',
      packages=['src', 'gui'],
      package_dir={'src': 'src/'},
      cmdclass={'build_ext': build_ext},
      include_package_data=True,
      zip_safe=False,
      install_requires=['numpy', 'scipy', 'mpi4py', 'h5py', 'matplotlib'],
      scripts=['scripts/DgaMain.py']
      )