# Overview
```mma4py``` is a parallel gradient-based optimizer that implements the method of moving asymptotes (MMA).
The algorithm is implemented in C++, and can be accessed from Python.
This project is based on [TopOpt_in_PETSc](https://github.com/topopt/TopOpt_in_PETSc), from which the source code of the implementation of MMA algorithm is obtained.

The original code can be found here: [https://github.com/topopt/TopOpt_in_PETSc](https://github.com/topopt/TopOpt_in_PETSc).
The corresponding publication is Aage, N., Andreassen, E., & Lazarov, B. S. (2015). Topology optimization using PETSc: An easy-to-use, fully parallel, open source topology optimization framework. Structural and Multidisciplinary Optimization, 51(3), 565–572. https://doi.org/10.1007/s00158-014-1157-0.


# How to use
To use ```mma4py```, simply define a custom class that implements the evaluation of objective, constraints and gradients, and feed the object to the optimizer.
An illustrative code snippet is shown below:

```python
from mma4py import Problem, Optimizer

class MyProb(Problem):
    def __init__(self, comm, nvars, nvars_l):
        ...

    def getVarsAndBounds(self, x, lb, ub):
        ...

    def evalObjCon(self, x, cons):
        ...

    def evalObjConGrad(self, x, g, gcon):
        ...

# Create problem instance
prob = MyProb(comm, nvars, nvars_l)

# Create optimization instance
opt = Optimizer(prob)

# Validate input gradients using finite difference
opt.checkGradients()

# Run optimization
opt.optimize(niter=100, verbose=True)

# Get the (distributed) optimized solution
xopt = opt.getOptimizedDesign()
```

See [examples/quadratic/quad-min.py](examples/quadratic/quad-min.py) for more details.

# Installation

- First, install python dependencies by ```pip install -r requirements.txt```
- Then, install [PETSc](https://petsc.org/release/):
    - Obtain source code from https://gitlab.com/petsc/petsc
    - Configure PETSc by ```./configure --with-mpi=1 --with-debugging=0 --prefix=dir/to/install/petsc```, then follow the steps in the print out
- Install ```mma4py```:
    - Specify location of PETSc by ```export MMA4PY_PETSC_PREFIX=dir/to/install/petsc```
    - In root directory of ```mma4py```, do ```pip install .```