# -*- python -*-
import os
import sys

from lsst.sconsUtils import env, scripts


def eigenInclude():
    """Return the directory holding ``Eigen/SparseCore`` (needed by src/sparseSolver.cc).

    Eigen is no longer an eups product in recent stacks; it is in the conda environment.
    Set EIGEN_INCLUDE_DIR to override.
    """
    e = os.environ
    candidates = [e.get("EIGEN_INCLUDE_DIR"),
                  os.path.join(e["EIGEN_DIR"], "include") if e.get("EIGEN_DIR") else None,
                  os.path.join(e["CONDA_PREFIX"], "include", "eigen3") if e.get("CONDA_PREFIX") else None,
                  os.path.join(sys.prefix, "include", "eigen3"),
                  "/usr/include/eigen3", "/usr/local/include/eigen3"]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "Eigen", "SparseCore")):
            return c
    raise RuntimeError("Eigen headers not found: set EIGEN_INCLUDE_DIR to the directory that "
                       "contains Eigen/SparseCore")


# Python-only packages would use scripts.BasicSConstruct("meas_crowd", disableCc=True)
scripts.BasicSConstruct.initialize("meas_crowd")
env.AppendUnique(CPPPATH=[eigenInclude()])
scripts.BasicSConstruct.finish()
