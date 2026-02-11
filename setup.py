from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import os
import subprocess

REQUIRES = ['numpy', 'scipy', 'matplotlib', 'h5py',
            'caput @ git+ssh://git@github.com/barawn/caput.git',      
            'healpy', 'astropy', 'mpi4py' ]

class CustomBuild_Py(build_py):
    def run(self):
        with open("install.log","w") as f:
            subprocess.run(["./compileNE2001.sh"], stdout=f)
        build_py.run(self)

setup(name='ULSA',
      version='0.1',
      description='The Ultral-Long wavelength Sky model with Absorption',
      cmdclass={"build_py":CustomBuild_Py},
      #long_description=readme(),
      install_requires=REQUIRES,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='The ULSA',
      url='https://github.com/Yanping-Cong/ULSA',
      author='Yanping Cong',
      author_email='ypcong@bao.ac.cn',
      license='MIT',
      #packages=['funniest/funniest'],
      packages = find_packages(),
      package_data={ 'ULSA' : [ 'ULSA/NE2001/NE2001_4python/NE2001_4python/src.NE2001/libNE2001.so' ] },
      #install_requires=[
      #    'markdown',
      #],
      include_package_data=True,
      zip_safe=False)
