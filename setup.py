# This file is part of meas_crowd.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Stand-alone build of the C++ extension, for use WITHOUT the Rubin stack (laptops, CI).

Inside the stack use ``setup -r . && scons`` instead.  Here::

    pip install pybind11
    python setup.py build_ext --inplace     # puts _crowdLib*.so into python/lsst/meas/crowd

The sparse joint-fit solver needs the Eigen headers (header-only).  They are found in, in order:
``$EIGEN_INCLUDE_DIR``, ``$EIGEN_DIR/include``, ``$CONDA_PREFIX/include/eigen3`` (the Rubin stack's
conda environment, so this works as-is on the RSP), ``/usr/include/eigen3`` and
``/usr/local/include/eigen3``.  Install them with ``conda install eigen`` or
``apt install libeigen3-dev`` if none is found.
"""
import os
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def findEigen():
    """Return the directory holding ``Eigen/SparseCore``, or exit with a hint."""
    env = os.environ
    candidates = [env.get("EIGEN_INCLUDE_DIR"),
                  os.path.join(env["EIGEN_DIR"], "include") if env.get("EIGEN_DIR") else None,
                  os.path.join(env["CONDA_PREFIX"], "include", "eigen3") if env.get("CONDA_PREFIX") else None,
                  os.path.join(sys.prefix, "include", "eigen3"),
                  "/usr/include/eigen3", "/usr/local/include/eigen3"]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "Eigen", "SparseCore")):
            return c
    sys.exit("Eigen headers not found: set EIGEN_INCLUDE_DIR, or `conda install eigen` / "
             "`apt install libeigen3-dev`")


ext = Pybind11Extension(
    "lsst.meas.crowd._crowdLib",
    ["python/lsst/meas/crowd/_crowdLib.cc", "src/stamps.cc", "src/sparseSolver.cc", "src/jointFit.cc",
     "src/centroids.cc"],
    include_dirs=["include", findEigen()],
    cxx_std=17,
    extra_compile_args=["-O3", "-DNDEBUG"],
)

setup(
    name="meas_crowd",
    license="GPL-3.0-or-later",
    package_dir={"": "python"},
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
