from __future__ import annotations
import numpy
from mpi4py import MPI

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
            c(x) >= 0
    """

    def __init__(
        self,
        comm: MPI.Comm,
        nvars: int,
        nvars_l: int,
        ncons: int,
    ) -> None:
        pass
    def evalObjCon(
        self,
        x: numpy.ndarray,
        cons: numpy.ndarray,
    ) -> float:
        pass
    def evalObjConGrad(
        self,
        x: numpy.ndarray,
        g: numpy.ndarray,
        gcon: numpy.ndarray,
    ) -> None: ...
    pass

class Optimizer:
    def __init__(self, prob: Problem) -> None: ...
    def optimize(self, niter: int) -> int: ...
    pass

def _petsc_initialize() -> None:
    pass

def _petsc_initialized() -> bool:
    pass
