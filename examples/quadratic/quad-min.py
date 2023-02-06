from mma4py import Problem, Optimizer
from mpi4py import MPI
import numpy as np


class Prob(Problem):
    def __init__(self, comm, nvars, nvars_l):
        super().__init__(comm, nvars, nvars_l, 1)
        self.comm = comm
        self.nvars = nvars
        self.nvars_l = nvars_l
        self.ncon = 1
        return

    def getVarsAndBounds(self, x, lb, ub):
        x[:] = 0.95
        lb[:] = 0.0
        ub[:] = 1.0
        return

    def evalObjCon(self, x, cons):
        _obj = np.array([np.sum(x**2)], dtype=float)
        _con = np.array([np.sum(x, dtype=float)])

        obj = np.zeros(1, dtype=float)

        self.comm.Allreduce(_con, cons)
        self.comm.Allreduce(_obj, obj)

        cons[0] = 1.0 - cons[0]
        return obj[0]

    def evalObjConGrad(self, x, g, gcon):
        g[:] = 2.0 * x[:]
        gcon[0, :] = -1.0
        return


comm = MPI.COMM_WORLD

nvars = 1000
nvars_l = nvars // comm.size
for i in range(nvars % comm.size):
    nvars_l += 1

prob = Prob(comm, nvars, nvars_l)
opt = Optimizer(prob, "mma4py.log")
opt.optimize(10)

xopt = opt.getOptimizedDesign()
print("%20.10e" % (np.sum(xopt)))
