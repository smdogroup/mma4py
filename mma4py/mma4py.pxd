
cimport petsc4py.PETSc as PETSc
from PETSc cimport PetscInt, PetsScalar, Vec

# For MPI capabilities
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

cdef extern from "MMA.h":
    cdef cppclass MMA:
        # Construct using defaults subproblem penalization
        MMA(MPI_Comm comm, PetscInt n, PetscInt m, Vec x)

        # Set the aggresivity of the moving asymptotes
        PetscErrorCode SetAsymptotes(PetscScalar init, PetscScalar decrease,
                                     PetscScalar increase)

        # val=0: default, val=1: increase robustness, i.e
        # control the spacing between L < alp < x < beta < U,
        PetscErrorCode SetRobustAsymptotesType(PetscInt val)

        # Set and solve a subproblem: return new xval
        PetscErrorCode Update(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx,
                              Vec xmin, Vec xmax)

        # Return necessary data for possible restart
        PetscErrorCode Restart(Vec xo1, Vec xo2, Vec U, Vec L)

        # Sets outer movelimits on all primal design variables
        # This is often requires to prevent the solver from oscilating
        PetscErrorCode SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax,
                                         PetscScalar movelim, Vec x, Vec xmin,
                                         Vec xmax)

        # Return KKT residual norms (norm2 and normInf)
        PetscErrorCode KKTresidual(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx,
                                   Vec xmin, Vec xmax, PetscScalar* norm2,
                                   PetscScalar* normInf)

        # Inf norm on diff between two vectors: SHOULD NOT BE HERE - USE BASIC
        # PETSc!!!!!
        PetscScalar DesignChange(Vec x, Vec xold)
