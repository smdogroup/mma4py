from mma4py.pywrapper import Problem, Optimizer
from mma4py.pywrapper import _petsc_initialize, _petsc_initialized

# Initialize petsc (which initializes MPI)
if not _petsc_initialized():
    _petsc_initialize()

# Affects from mma4py import *
__all__ = ["Problem", "Optimizer"]
