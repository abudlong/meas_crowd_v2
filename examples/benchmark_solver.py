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

"""Time the joint-fit solvers on synthetic crowded tiles.

Compares pipe_crowd's sparse Eigen LSCG solver (``jointSolver="eigen"``) with the previous dense
normal-equation NNLS (``"normal"``) and ``scipy.optimize.lsq_linear`` (``"scipy"``)::

    setup -k -r /path/to/meas_crowd   (or PYTHONPATH=/path/to/meas_crowd/python)
    python examples/benchmark_solver.py                       # 300 ... 4000 sources
    python examples/benchmark_solver.py --sizes 1000 8000 --solvers eigen normal

Without the Rubin stack, run it from ``tests/`` (or with ``tests`` on PYTHONPATH) so the stack
stand-ins are found; the "normal" solver lives in the pipeline module, which imports the stack.
"""
import argparse
import sys


def main(argv=None):
    """Run the benchmark."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 1000, 2000, 4000])
    ap.add_argument("--solvers", nargs="+", default=["eigen", "normal", "scipy"])
    ap.add_argument("--scipy-max", type=int, default=1000,
                    help="largest problem given to scipy's lsq_linear (it is slow)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        import lsst.afw  # noqa: F401
    except ImportError:
        try:
            import lsstStubs
            lsstStubs.install()
        except ImportError:
            pass
    from lsst.meas.crowd.sparseSolver import compareSolvers, makeCrowdedFieldProblem
    for n in args.sizes:
        A, b, _ = makeCrowdedFieldProblem(nSources=n, seed=args.seed)
        names = tuple(s for s in args.solvers if s != "scipy" or n <= args.scipy_max)
        compareSolvers(A, b, solvers=names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
