# ULSA

## Ultra Long wavelength Sky model with Absorption(ULSA)
--------
The ULSA implementation of the NE2001 electron model using in 3D emissivity of galaxy to produce a sky map below 10MHz.

The output can be stored either in memory by means of a numpy array, or in a HDF5 format file.

## Updated Installation (PSA/cozzyd)

You need a Fortran compiler (something that implements f77 - gfortran is fine).
You need an MPI implementation installed (e.g. libmpich or OpenMPI - see <https://github.com/mpi4py/mpi4py>. You probably want OpenMPI.

You probably want a Python virtual environment so you might need the ``venv`` Python package installed.
Otherwise be bold and ignore the first two lines.

```
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install .
```

This creates a ``venv`` in the ``venv`` subdirectory. If you don't want it there, just change that last
venv and also change the source line.

## Example (updated PSA/cozzyd)

Look at example/example.py. You probably want to run this using ``mpirun`` to take advantage of multiple
cores on your system (e.g. ``mpirun python3 example.py``).

NOTE: The NE2001 input files (everything ending in .inp or .dat) in that directory
are required to run any ULSA program. They have to be located in that directory.

## Documentation
Documentation can be found at <https://ulsa.readthedocs.io/en/latest/>.

## Warning
If this code helps you, please refer to the [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4454153.svg)](https://doi.org/10.5281/zenodo.4454153), or contact me by email and I will send you a link to the article ypcong@bao.ac.cn

# Updated Instructions (PSA/cozzyd)


