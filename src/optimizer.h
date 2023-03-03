#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "MMA.h"
#include "mpi4py/mpi4py.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

using ndarray_t = pybind11::array_t<double, pybind11::array::c_style>;

/**
 * @brief Helper function: allocate the petsc vector x
 *
 * @param x vector
 * @param comm MPI global communicator
 * @param gsize global vector size
 * @param lsize local vector size, determined by petsc by default
 * @return PetscErrorCode
 */
PetscErrorCode allocate_petsc_vec(Vec* x, const MPI_Comm comm,
                                  const PetscInt gsize, const PetscInt lsize);

/**
 * @brief Helper function: bind a petsc vector with provided memory
 *
 * @param x vector
 * @param comm MPI global communicator
 * @param gsize global vector size
 * @param lsize local vector size, determined by petsc by default
 * @param data pointer to user managed data
 * @return PetscErrorCode
 */
PetscErrorCode bind_petsc_vec_to_array(Vec* x, const MPI_Comm comm,
                                       const PetscInt gsize,
                                       const PetscInt lsize, PetscScalar* data);

/**
 * @brief An abstract optimization problem class, to perform optimization,
 * user needs to implement the following methods:
 *
 *  - getVarsAndBounds()
 *  - evalObjCon()
 *  - evalObjConGrad()
 *
 * Note1: we assume that the local partitions of global dv, obj gradient
 * and constraint gradients are partitioned in the same way, i.e. local
 * sizes of these vectors are the same.
 *
 * Note2: Assume constraints take the following form:
 *        c(x) <= 0
 */
class Problem {
 public:
  /**
   * @param nvars global design variable vector size
   * @param nvars_l local design variable vector size
   * @param ncons number of constraints
   */
  Problem(pybind11::object py_comm, int nvars, int nvars_l, int ncons)
      : comm(*get_mpi_comm(py_comm)),
        nvars(nvars),
        nvars_l(nvars_l),
        ncons(ncons){};
  Problem(int nvars, int nvars_l, int ncons)
      : nvars(nvars), nvars_l(nvars_l), ncons(ncons) {
    comm = MPI_COMM_SELF;
  };

  /**
   * @brief Destructor
   *
   * Note: Define it as virtual as we might need to delete object using base
   * type pointer
   */
  virtual ~Problem() = default;

  /**
   * @brief Set initial design x and its lower/upper bounds
   *
   * @param x0 initial design
   * @param lb lower bound
   * @param ub upper bound
   */
  virtual void getVarsAndBounds(ndarray_t x0, ndarray_t lb, ndarray_t ub) = 0;

  /**
   * @brief Evaluate objective and constraints
   *
   * @param x design variable, 1d array
   * @param cons constraints with the form c(x) <= 0, 1d array
   * @return objective
   */
  virtual double evalObjCon(ndarray_t x, ndarray_t cons) = 0;

  /**
   * @brief Evaluate objective and constraint gradients
   *
   * @param x design variable, 1d array
   * @param g objective gradient, 1d array
   * @param gcon constraint gradients, 2d array, gcon[i, :] for i-th
   * constraint
   */
  virtual void evalObjConGrad(ndarray_t x, ndarray_t g, ndarray_t gcon) = 0;

  inline const MPI_Comm get_mpi_comm() const { return comm; }
  inline const int get_num_cons() const { return ncons; }
  inline const int get_num_vars() const { return nvars; }
  inline const int get_num_vars_local() const { return nvars_l; }

  /**
   * @brief Helper function: convert python MPI comm to MPI_Comm
   *
   * @param mpi4py_comm
   */
  static inline MPI_Comm* get_mpi_comm(pybind11::object mpi4py_comm) {
    auto comm_ptr = PyMPIComm_Get(mpi4py_comm.ptr());
    if (!comm_ptr) {
      throw pybind11::error_already_set();
    }
    return comm_ptr;
  }

 protected:
  MPI_Comm comm;
  int nvars;    // global number of dvs
  int nvars_l;  // local number of dvs
  int ncons;    // number of constraints
};

/**
 * @brief The optimizer
 */
class Optimizer final {
 public:
  Optimizer(Problem* prob, const char* log_name);
  ~Optimizer();

  /**
   * @brief Check the gradients by finite differencing
   *
   * @param h finite difference step
   */
  void checkGradients(unsigned int seed = 0u, double h = 1e-6);

  /**
   * @brief Perform optimization
   *
   * @param niter number of iterations
   * @param verbose print optimization history to stdout or not
   */
  PetscErrorCode optimize(int niter, bool verbose = false);

  /**
   * @brief Return xopt
   *
   * @return ndarray_t xopt
   */
  ndarray_t getOptimizedDesign();

 private:
  Problem* prob;
  const char* log_name;
  FILE* fp;
  double obj;  // objective value
  Vec x, g;    // design variable and objective gradient
  Vec* gcon;   // constraint function gradients
  Vec lb, ub;  // upper and lower bounds of x

  ndarray_t np_x, np_cons, np_g, np_gcon, np_lb, np_ub;  // numpy arrays
};

#endif