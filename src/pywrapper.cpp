#include <mpi.h>
#include <mpi4py/mpi4py.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

#include "MMA.h"
#include "optimizer.h"

namespace py = pybind11;

/**
 * @brief Helper class for defining inherited class from python
 */
class PyProblem : public Problem {
 public:
  // Inherit constructor from base class
  using Problem::Problem;

  // Trampoline for pure virtual functions
  double evalObjCon(ndarray_t x, ndarray_t cons) override {
    PYBIND11_OVERRIDE_PURE(double,      // return type
                           Problem,  // Parent class
                           evalObjCon,  // name of function
                           x, cons      // Argument(s), if any
    );
  }

  void evalObjConGrad(ndarray_t x, ndarray_t g, ndarray_t gcon) override {
    PYBIND11_OVERRIDE_PURE(void,            // return type
                           Problem,      // Parent class
                           evalObjConGrad,  // name of function
                           x, g, gcon       // Argument(s), if any
    );
  }
};

void initialize(py::object py_comm) {
  MPI_Comm comm = *Problem::get_mpi_comm(py_comm);
  PetscCallAbort(comm, PetscInitialize(nullptr, nullptr, nullptr, nullptr));
  return;
}

PYBIND11_MODULE(pywrapper, m) {
  // initialize mpi4py's C-API
  if (import_mpi4py() < 0) {
    // mpi4py calls the Python C API
    // we let pybind11 give us the detailed traceback
    throw py::error_already_set();
  }

  py::class_<Problem, PyProblem>(m, "Problem")
      .def(py::init<py::object, int, int, int>())
      .def("evalObjCon", &Problem::evalObjCon)
      .def("evalObjConGrad", &Problem::evalObjConGrad);

  py::class_<Optimizer>(m, "Optimizer")
      .def(py::init<Problem*>())
      .def("optimize", &Optimizer::optimize);

  m.def("initialize", &initialize);
}
