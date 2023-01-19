# distutils: language=c++

from mma4py cimport *

# Import numpy
cimport numpy as np
import numpy as np

# For MPI capabilities
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

# Include the mpi4py header
cdef extern from "mpi-compat.h":
    pass

cdef class mma:
    cdef MMA *ptr

    def __cinit__(self, comm, n, m, x):
        self.ptr = new MMA(comm, n, m, x)

    def update(self, x, dfdx, g, dgdx, xmin, xmax):

    def set_outer_move_limit(self, Xmin, Xmax, movelim, x, xmin, xmax):


