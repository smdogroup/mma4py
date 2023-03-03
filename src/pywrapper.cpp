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
  void getVarsAndBounds(ndarray_t x, ndarray_t lb, ndarray_t ub) override {
    PYBIND11_OVERRIDE_PURE(void,              // return type
                           Problem,           // Parent class
                           getVarsAndBounds,  // name of function
                           x, lb, ub          // Argument(s), if any
    );
  }

  double evalObjCon(ndarray_t x, ndarray_t cons) override {
    PYBIND11_OVERRIDE_PURE(double,      // return type
                           Problem,     // Parent class
                           evalObjCon,  // name of function
                           x, cons      // Argument(s), if any
    );
  }

  void evalObjConGrad(ndarray_t x, ndarray_t g, ndarray_t gcon) override {
    PYBIND11_OVERRIDE_PURE(void,            // return type
                           Problem,         // Parent class
                           evalObjConGrad,  // name of function
                           x, g, gcon       // Argument(s), if any
    );
  }
};

void _petsc_initialize() {
  PetscInitialize(nullptr, nullptr, nullptr, nullptr);
  return;
}

bool _petsc_initialized() {
  PetscBool initialized;
  PetscInitialized(&initialized);
  return (bool)initialized;
}

PYBIND11_MODULE(pywrapper, m) {
  // initialize mpi4py's C-API
  if (import_mpi4py() < 0) {
    // mpi4py calls the Python C API
    // we let pybind11 give us the detailed traceback
    throw py::error_already_set();
  }

  py::class_<Problem, PyProblem>(m, "Problem")
      .def(py::init<py::object, int, int, int>(), py::arg("comm"),
           py::arg("nvars"), py::arg("nvars_l"), py::arg("ncons"))
      .def(py::init<int, int, int>(), py::arg("nvars"), py::arg("nvars_l"),
           py::arg("ncons"))
      .def("getVarsAndBounds", &Problem::getVarsAndBounds, py::arg("x0"),
           py::arg("lb"), py::arg("ub"))
      .def("evalObjCon", &Problem::evalObjCon, py::arg("x"), py::arg("cons"))
      .def("evalObjConGrad", &Problem::evalObjConGrad, py::arg("x"),
           py::arg("g"), py::arg("gcon"));

  py::class_<Optimizer>(m, "Optimizer")
      .def(py::init<Problem*, const char*>(), py::arg("prob"),
           py::arg("log_name") = "mma4py.log")
      .def("checkGradients", &Optimizer::checkGradients, py::arg("seed") = 0u,
           py::arg("h") = 1e-6)
      .def("optimize", &Optimizer::optimize, py::arg("niter"),
           py::arg("verbose") = false)
      .def("getOptimizedDesign", &Optimizer::getOptimizedDesign);

  m.def("_petsc_initialize", &_petsc_initialize);
  m.def("_petsc_initialized", &_petsc_initialized);
}
