from __future__ import annotations
import numpy

__all__ = ["Optimizer", "Problem"]

class Problem:
    """
    An abstract optimization problem class, to perform optimization,
    user needs to implement the following methods:

    - evalObjCon()
    - evalObjConGrad()

    Note1: we assume that the local partitions of global dv, obj gradient
    and constraint gradients are partitioned in the same way, i.e. local
    sizes of these vectors are the same.

    Note2: Assume constraints take the following form:
            c(x) <= 0
    """

    def __init__(self, comm, nvars, nvars_l, ncons) -> None:
        """
        Initializer.

        Args:
            comm: MPI.Comm
            nvars: int, size of the global design vector x
            nvars_l: int, size of the local portion of x at current MPI processor
            ncons: number of constraints
        """
        pass
    def getVarsAndBounds(self, x0, lb, ub) -> None:
        """
        Set initial design, lower and upper bounds.

        Args:
            x0: 1d numpy array, initial design vector
            lb: 1d numpy array, lower bounds
            ub: 1d numpy array, upper bounds
        """
        pass
    def evalObjCon(self, x, cons) -> float:
        """
        Evaluate objective and constraints.

        Args:
            x: 1d numpy array, contains design variable at current optimization
               iteration
            cons: 1d numpy array, constraint values to be updated from user's
                  implementation

        Return:
            objective value
        """
        pass
    def evalObjConGrad(self, x, g, gcon) -> None:
        """
        Evaluate objective and constraint gradients.

        Args:
            x: 1d numpy array, contains design variable at current optimization
               iteration
            g: 1d numpy array, objective gradient to be updated from user's
                  implementation
            gcon: 2d numpy array, gcon[i, :] contains gradient of i-th
                  constraint, to be updated from user's implementation
        """
        pass

class Optimizer:
    def __init__(self, prob, log_name="mma4py.log") -> None:
        """
        Initialize MMA optimizer.

        Args:
            prob: Problem, the problem instance
            log_name: string, output log file name
        """
        pass
    def checkGradients(self, seed=0, h=1e-6) -> None:
        """
        Check the gradient using finite difference.

        Args:
            seed: int, seed for random number generator
            h: finite difference step
        """
        pass
    def optimize(self, niter, verbose=False) -> int:
        """
        Execute optimization.

        Args:
            niter: int, number of iterations

        Return:
            error code, 0 is success
        """
        pass
    def getOptimizedDesign(self) -> numpy.ndarray:
        """
        Get optimized design variable xopt.

        Return:
            1d numpy array, optimized x
        """
    pass

def _petsc_initialize() -> None:
    pass

def _petsc_initialized() -> bool:
    pass
