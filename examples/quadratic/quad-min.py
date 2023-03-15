from mma4py import Problem, Optimizer
from mpi4py import MPI
import numpy as np
import argparse


class Prob(Problem):
    def __init__(self, comm, nvars, nvars_l):
        self.comm = comm
        self.nvars = nvars
        self.nvars_l = nvars_l
        self.ncon = 2
        super().__init__(
            comm=self.comm, nvars=self.nvars, nvars_l=self.nvars_l, ncons=self.ncon
        )

        np.random.seed(0)
        self.w1 = np.random.rand(self.nvars_l)
        self.w2 = np.random.rand(self.nvars_l)
        return

    def getVarsAndBounds(self, x, lb, ub):
        x[:] = 0.95
        lb[:] = 0.0
        ub[:] = 1.0
        return

    def evalObjCon(self, x, cons):
        _obj = np.array([np.sum(x**2)], dtype=float)
        _con = np.array(
            [np.sum(x * self.w1, dtype=float), np.sum(x * self.w2, dtype=float)]
        )

        obj = np.zeros(1, dtype=float)

        self.comm.Allreduce(_con, cons)
        self.comm.Allreduce(_obj, obj)

        cons[:] = 1.0 - cons[:]
        return obj[0]

    def evalObjConGrad(self, x, g, gcon):
        g[:] = 2.0 * x[:]
        gcon[0, :] = -self.w1
        gcon[1, :] = -self.w2
        return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--maxit", default=200, type=int)
    p.add_argument("--movelim", default=0.1, type=float)
    p.add_argument("--tol", default=1e-8, type=float)
    args = p.parse_args()

    # MPI communicator
    comm = MPI.COMM_WORLD

    # Hard-code global vector size and compute sizes of local portions at each
    # processor
    nvars = 1000
    nvars_l = nvars // comm.size
    if comm.rank < nvars % comm.size:
        nvars_l += 1

    # Create problem instance
    prob = Prob(comm, nvars, nvars_l)

    # Create optimization instance
    opt = Optimizer(prob)

    # Check consistency of input gradients using finite difference
    opt.checkGradients(seed=0, h=1e-6)

    # Run optimization
    opt.optimize(
        niter=args.maxit,
        verbose=True,
        movelim=args.movelim,
        atol_l2=args.tol,
        atol_linf=args.tol,
    )

    # Get distributed optimized solution
    xopt = opt.getOptimizedDesign()
    flag = opt.getSuccessFlag()

    # Perform MPI communication to obtain the global solution vector
    xopt_g = np.empty(nvars)
    comm.Allgatherv(xopt, xopt_g)
    if comm.rank == 0:
        print("success flag:", flag)
        print("\nOptimized solution:\n|xopt|_1: %20.10e" % (np.sum(xopt_g)))
