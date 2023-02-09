from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import mpi4py
from glob import glob
from subprocess import check_output
from os.path import join, realpath
from petsc import get_petsc_dir


def get_petsc_dirs():
    # Find petsc
    petsc_prefix = get_petsc_dir()

    petsc_inc_dir = realpath(join(petsc_prefix, "include"))
    petsc_lib_dir = realpath(join(petsc_prefix, "lib"))

    return petsc_inc_dir, petsc_lib_dir


def get_mpi_flags(mpiexec="mpicxx"):
    # Split the output from the mpicxx command
    args = check_output([mpiexec, "-show"]).decode("utf-8").split()

    # Determine whether the output is an include/link/lib command
    inc_dirs, lib_dirs, libs = [], [], []
    for flag in args:
        if flag[:2] == "-I":
            inc_dirs.append(flag[2:])
        elif flag[:2] == "-L":
            lib_dirs.append(flag[2:])
        elif flag[:2] == "-l":
            libs.append(flag[2:])

    return inc_dirs, lib_dirs, libs


if __name__ == "__main__":
    petsc_inc_dir, petsc_lib_dir = get_petsc_dirs()
    mpi_inc_dirs, mpi_lib_dirs, mpi_libs = get_mpi_flags()

    include_dirs = [
        petsc_inc_dir,
        *mpi_inc_dirs,
        mpi4py.get_include(),
    ]

    library_dirs = [*mpi_lib_dirs, petsc_lib_dir]

    ext_modules = [
        Pybind11Extension(
            "pywrapper",
            glob("src/*.cpp"),
            cxx_std=11,
            library_dirs=library_dirs,
            libraries=[*mpi_libs, "petsc"],
        ),
    ]

    setup(
        name="mma4py",
        version="0.1.0",
        description="a parallel MMA optimizer with python wrapper",
        url="https://github.com/aaronyicongfu/mma4py",
        author="Yicong Fu",
        author_email="fuyicong1996@gmail.com",
        license="Apache-2.0 license",
        packages=["mma4py"],
        ext_package="mma4py",
        ext_modules=ext_modules,
        include_dirs=include_dirs,
        # install_requires=["pybind11", "mpi4py", "petsc"],
    )
