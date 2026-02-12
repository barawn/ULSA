#!/bin/bash

cd ULSA/NE2001/NE2001_4python/NE2001_4python/src.NE2001
make clean
if [ -x "$(command -v ifort)" ]; then
    export FC=ifort
    export FFLAGS=-O
elif [ -x "$(command -v gfortran)" ]; then
    export FC=gfortran
    export FFLAGS="-O -std=legacy"
elif [ -x "$(command -v f77)" ]; then
    export FC=f77
    export FFLAGS=-O
fi
echo FC=$FC FFLAGS=$FFLAGS make so
FC=$FC FFLAGS="$FFLAGS" make so
