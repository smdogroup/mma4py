#include "optimizer.h"

PetscErrorCode allocate_petsc_vec(Vec* x, const MPI_Comm comm,
                                  const PetscInt gsize, const PetscInt lsize) {
  // Create an empty vector object
  PetscCall(VecCreate(comm, &(*x)));

  // Set global and optionally local size
  PetscCall(VecSetSizes(*x, lsize, gsize));

  // This vector is an MPI vector
  PetscCall(VecSetType(*x, VECMPI));

  return 0;
}

PetscErrorCode bind_petsc_vec_to_array(Vec* x, const MPI_Comm comm,
                                       const PetscInt gsize,
                                       const PetscInt lsize,
                                       PetscScalar* data) {
  // Create an empty vector object
  PetscCall(VecCreate(comm, &(*x)));

  // Set global and optionally local size
  PetscCall(VecSetSizes(*x, lsize, gsize));

  // This vector is an MPI vector
  PetscCall(VecSetType(*x, VECMPI));

  // Use user provided memory
  PetscCall(VecPlaceArray(*x, data));

  return 0;
}

Optimizer::Optimizer(Problem* prob) : prob(prob) {
  // Get meta data
  MPI_Comm comm = prob->get_mpi_comm();
  int nvars = prob->get_num_vars();
  int nvars_l = prob->get_num_vars_local();
  int ncons = prob->get_num_cons();

  // Allocate numpy arrays
  np_x = ndarray_t(nvars_l);
  np_g = ndarray_t(nvars_l);
  np_cons = ndarray_t(ncons);
  np_gcon = ndarray_t(ncons * nvars_l);
  np_gcon = np_gcon.reshape({ncons, nvars_l});

  // Allocate design variable and gradient vectors
  PetscCallAbort(comm, bind_petsc_vec_to_array(&x, comm, nvars, nvars_l,
                                               (double*)np_x.data()));
  PetscCallAbort(comm, bind_petsc_vec_to_array(&g, comm, nvars, nvars_l,
                                               (double*)np_g.data()));

  gcon = new Vec[ncons];

  double* gcon_data = (double*)np_gcon.data();
  int offset = 0;
  for (PetscInt i = 0; i < ncons; i++) {
    PetscCallAbort(comm, bind_petsc_vec_to_array(&gcon[i], comm, nvars, nvars_l,
                                                 &gcon_data[offset]));
    offset += nvars_l;
  }
}

Optimizer::~Optimizer() {
  MPI_Comm comm = prob->get_mpi_comm();
  int ncons = prob->get_num_cons();

  PetscCallAbort(comm, VecDestroy(&x));
  PetscCallAbort(comm, VecDestroy(&g));
  for (PetscInt i = 0; i < ncons; i++) {
    PetscCallAbort(comm, VecDestroy(&gcon[i]));
  }

  delete[] gcon;
}

PetscErrorCode Optimizer::optimize(int niter) {
  MPI_Comm comm = prob->get_mpi_comm();
  int nvars = prob->get_num_vars();
  int nvars_l = prob->get_num_vars_local();
  int ncons = prob->get_num_cons();

  // Set parameters
  double movelim = 0.2;
  double lb = 0.0, ub = 1.0, x0 = 0.0;

  // Set initial design
  PetscCall(VecSet(x, x0));

  // Allocate and initialize bounds
  Vec lbvec, ubvec;
  PetscCall(allocate_petsc_vec(&lbvec, comm, nvars, nvars_l));
  PetscCall(allocate_petsc_vec(&ubvec, comm, nvars, nvars_l));
  PetscCall(VecSet(lbvec, lb));
  PetscCall(VecSet(ubvec, ub));

  // Allocate mma operator
  MMA mma(nvars, ncons, x);

  // Optimization loop body
  int iter = 0;
  while (iter < niter) {
    // Get arrays associated with PETSc vectors
    // PetscScalar *xvals, *gvals;
    // PetscCall(VecGetArray(x, &xvals));
    // PetscCall(VecGetArray(g, &gvals));

    // for (int i = 0; i < ncons; i++) {
    //   PetscCall(VecGetArray(gcon[i], &gconvals[i]));
    // }

    // Evaluate functions and gradients
    obj = prob->evalObjCon(np_x, np_cons);
    prob->evalObjConGrad(np_x, np_g, np_gcon);

    // // Done modifying underlying arrays of PETSc vectors
    // PetscCall(VecRestoreArray(x, &xvals));
    // PetscCall(VecRestoreArray(g, &gvals));

    // for (int i = 0; i < ncons; i++) {
    //   PetscCall(VecRestoreArray(gcon[i], &gconvals[i]));
    // }

    // Set move limits
    PetscCall(mma.SetOuterMovelimit(lb, ub, movelim, x, lbvec, ubvec));

    // Update design
    mma.Update(x, g, (double*)np_cons.data(), gcon, lbvec, ubvec);

    // Check KKT error
    PetscScalar kkterr_l2, kkterr_linf;
    mma.KKTresidual(x, g, (double*)np_cons.data(), gcon, lbvec, ubvec,
                    &kkterr_l2, &kkterr_linf);

    // Compute dv l1 norm
    PetscScalar x_l1;
    PetscCall(VecNorm(x, NORM_1, &x_l1));

    // Print out
    if (iter % 10 == 0) {
      PetscCall(PetscPrintf(comm, "\n%6s%20s%20s%20s%20s\n", "iter", "obj",
                            "KKT_l2", "KKT_linf", "|x|_1"));
    }
    PetscCall(PetscPrintf(comm, "%6d%20.10e%20.10e%20.10e%20.12e\n", iter, obj,
                          kkterr_l2, kkterr_linf, x_l1));

    iter++;
  }

  PetscCall(VecDestroy(&lbvec));
  PetscCall(VecDestroy(&ubvec));

  return 0;
}