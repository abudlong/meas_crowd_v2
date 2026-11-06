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

"""Sparse joint-fit solver: pipe_crowd's Eigen LSCG solve, with a non-negativity bound.

lsst-dm/pipe_crowd solves the crowded-field joint fit by building the sparse (pixels x sources)
design matrix and handing it to ``Eigen::LeastSquaresConjugateGradient`` (conjugate gradients on
the normal equations with a Jacobi preconditioner), so ``A^T A`` is never formed and every
iteration costs two sparse matrix-vector products.  `solveSparse` does the same solve in C++
(``src/sparseSolver.cc``), wrapped in an active-set loop that keeps meas_crowd's x >= 0 bound:

1. solve on the free columns with LSCG (warm-started from the current x);
2. columns that came out negative are fixed at zero and step 1 repeated;
3. once all free values are >= 0, fixed columns whose gradient ``A_j^T (b - A x)`` is positive
   (the fit wants them above zero) are freed and step 1 repeated; otherwise done.

The pipeline uses it for every joint fit when ``jointSolver="eigen"`` (the default); see
`lsst.meas.crowd.pipeline.setJointSolver`.
"""

__all__ = ["HAVE_SPARSE_SOLVER", "solveSparse", "SparseSolveResult", "makeCrowdedFieldProblem",
           "compareSolvers"]

from dataclasses import dataclass

import time

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, issparse

from ._cxx import load as _loadCxx

_solve = getattr(_loadCxx(), "solveSparseLeastSquares", None)   # raises if the build is stale
HAVE_SPARSE_SOLVER = _solve is not None


@dataclass
class SparseSolveResult:
    """Result of `solveSparse`.

    Attributes
    ----------
    x : `numpy.ndarray`
        Solution, one value per column of A.
    cgIterations : `int`
        Conjugate-gradient iterations summed over every LSCG solve.
    activeSetIterations : `int`
        Number of LSCG solves (1 without the bound).
    nAtBound : `int`
        Columns held at zero by the bound.
    converged : `bool`
        Every LSCG solve converged and the bounded problem's optimality conditions hold.
    cgError : `float`
        Relative normal-equation residual ``||A^T r|| / ||A^T b||`` of the last solve.
    """

    x: np.ndarray
    cgIterations: int
    activeSetIterations: int
    nAtBound: int
    converged: bool
    cgError: float


def solveSparse(A, b, nonNegative=True, x0=None, tol=1e-8, maxIter=5000, maxActiveSet=50,
                kktTol=1e-6, unbounded=None):
    """Solve ``min ||A x - b||^2`` (optionally with ``x >= 0``) with Eigen's sparse LSCG.

    Parameters
    ----------
    A : `scipy.sparse.spmatrix` or `numpy.ndarray`
        Design matrix, rows = (weighted) pixels, columns = parameters.  Converted to CSC.
    b : `numpy.ndarray`
        Right-hand side (weighted data), one value per row.
    nonNegative : `bool`, optional
        Enforce ``x >= 0`` (the active-set loop); False gives pipe_crowd's unconstrained solve.
    x0 : `numpy.ndarray`, optional
        Starting guess (e.g. the previous iteration's fluxes); zeros if None.
    tol : `float`, optional
        LSCG stopping tolerance on ``||A^T r|| / ||A^T b||`` (pipe_crowd uses 1e-6).
    maxIter : `int`, optional
        LSCG iteration cap per solve (pipe_crowd uses 500).
    maxActiveSet : `int`, optional
        Cap on the number of LSCG solves in the active-set loop.
    kktTol : `float`, optional
        A column held at zero is freed when its gradient exceeds this fraction of
        ``max |A^T b|``.
    unbounded : `numpy.ndarray` [`bool`], optional
        Columns exempt from the ``x >= 0`` bound (e.g. sky terms); None = all bounded.

    Returns
    -------
    result : `SparseSolveResult`
        Solution and solver statistics.

    Raises
    ------
    ImportError
        If the C++ extension was built without the solver (rebuild it; see the README).
    """
    if _solve is None:
        raise ImportError("lsst.meas.crowd._crowdLib has no sparse solver: rebuild the extension "
                          "(`scons`, or `python setup.py build_ext --inplace`; needs Eigen)")
    A = csc_matrix(A) if not issparse(A) else A.tocsc()
    A.sum_duplicates()
    if A.nnz >= 2 ** 31:
        raise ValueError(f"matrix has {A.nnz} entries; the solver uses 32-bit indices")
    b = np.ascontiguousarray(b, dtype=float).ravel()
    nRows, nCols = A.shape
    if x0 is not None:
        x0 = np.ascontiguousarray(x0, dtype=float).ravel()
    res = _solve(nRows, nCols, A.indptr.astype(np.int32, copy=False),
                 A.indices.astype(np.int32, copy=False),
                 np.ascontiguousarray(A.data, dtype=float), b, x0, bool(nonNegative),
                 float(tol), int(maxIter), int(maxActiveSet), float(kktTol),
                 None if unbounded is None else np.ascontiguousarray(unbounded, dtype=np.uint8))
    return SparseSolveResult(**res)


def _psfKernel(half=40, fwhm=3.8, wingFrac=0.08, beta=2.5):
    """Unit-sum PSF: Gaussian core plus a Moffat wing, (2 half + 1) px square."""
    u = np.arange(-half, half + 1)
    r2 = u[None, :] ** 2 + u[:, None] ** 2
    sig = fwhm / 2.3548
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    core = np.exp(-0.5 * r2 / sig ** 2)
    wing = (1 + r2 / alpha ** 2) ** -beta
    k = (1 - wingFrac) * core / core.sum() + wingFrac * wing / wing.sum()
    return k / k.sum()


def makeCrowdedFieldProblem(nSources=2000, size=460, fwhm=3.8, sky=400.0, seed=0,
                            floorSigma=0.05, half=40):
    """Make a synthetic crowded-field joint fit shaped like `fitLib.jointPsfFitDiag`'s.

    Point sources with a power-law flux distribution are placed at random in a ``size`` x
    ``size`` image (default: a 400 px tile with 30 px padding) with an 81 x 81 px PSF; each
    source's stamp is cropped where its wing falls below ``floorSigma`` sigma per pixel (the
    pipeline's ``stampFloorSigma``), and rows are weighted by 1 / sigma.

    Parameters
    ----------
    nSources : `int`, optional
        Number of sources.
    size : `int`, optional
        Image side [px].
    fwhm : `float`, optional
        PSF FWHM [px].
    sky : `float`, optional
        Sky variance per pixel.
    seed : `int`, optional
        Random seed.
    floorSigma : `float`, optional
        Stamp crop level in sigma per pixel.
    half : `int`, optional
        Half-size of the full PSF stamp [px].

    Returns
    -------
    A : `scipy.sparse.csr_matrix`
        Weighted design matrix (pixels x sources).
    b : `numpy.ndarray`
        Weighted data.
    truth : `numpy.ndarray`
        True fluxes.
    """
    rng = np.random.default_rng(seed)
    K = _psfKernel(half, fwhm)
    sigma = np.sqrt(sky)
    fmin, fmax = 5 * sigma / K.max(), 2e6
    flux = fmin * (1 - rng.uniform(size=nSources)) ** (-1 / 0.6)
    flux = np.minimum(flux, fmax)
    xy = rng.uniform(-3, size + 2, size=(nSources, 2))
    u = np.arange(-half, half + 1)
    rows, cols, vals = [], [], []
    model = np.zeros((size, size))
    radial = np.array([K[half, half + r] for r in range(half + 1)])
    for j, ((x, y), f) in enumerate(zip(xy, flux)):
        ix, iy = int(round(x)), int(round(y))
        # sub-pixel shift by evaluating the analytic profile at the offset grid
        dx, dy = x - ix, y - iy
        r2 = (u[None, :] - dx) ** 2 + (u[:, None] - dy) ** 2
        sig = fwhm / 2.3548
        alpha = fwhm / (2 * np.sqrt(2 ** (1 / 2.5) - 1))
        st = 0.92 * np.exp(-0.5 * r2 / sig ** 2) / (2 * np.pi * sig ** 2) \
            + 0.08 * (1 + r2 / alpha ** 2) ** -2.5 * (1.5 / (np.pi * alpha ** 2))
        h = int(np.searchsorted(-radial * f, -floorSigma * sigma)) if floorSigma else half
        h = int(np.clip(h, 3, half))
        st = st[half - h:half + h + 1, half - h:half + h + 1]
        ya, yb, xa, xb = iy - h, iy + h + 1, ix - h, ix + h + 1
        cy0, cx0 = max(0, -ya), max(0, -xa)
        ya, xa, yb, xb = max(ya, 0), max(xa, 0), min(yb, size), min(xb, size)
        if ya >= yb or xa >= xb:
            continue
        sub = st[cy0:cy0 + yb - ya, cx0:cx0 + xb - xa]
        model[ya:yb, xa:xb] += f * sub
        yy, xx = np.mgrid[ya:yb, xa:xb]
        rows.append((yy * size + xx).ravel())
        cols.append(np.full(sub.size, j))
        vals.append(sub.ravel())
    var = sky + model
    data = model + rng.normal(size=model.shape) * np.sqrt(var)
    w = 1 / np.sqrt(var.ravel())
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(size * size, nSources)).tocsr()
    A = A.multiply(w[:, None]).tocsr()
    return A, data.ravel() * w, flux


def compareSolvers(A, b, solvers=("eigen", "normal", "scipy"), verbose=True):
    """Time the joint-fit solvers on one problem and compare their answers.

    Parameters
    ----------
    A : `scipy.sparse.spmatrix`
        Weighted design matrix.
    b : `numpy.ndarray`
        Weighted data.
    solvers : `tuple` [`str`], optional
        Any of ``"eigen"`` (`solveSparse`), ``"normal"`` (meas_crowd's dense normal-equation
        NNLS, `pipeline._nnlsNormal`) and ``"scipy"`` (``scipy.optimize.lsq_linear``, method
        'trf', as `fitLib.jointPsfFitDiag` calls it).
    verbose : `bool`, optional
        Print a table.

    Returns
    -------
    results : `dict`
        Per solver: ``seconds``, ``x``, ``cost`` (0.5 ||Ax - b||^2) and ``maxRelDiff`` (largest
        flux difference to the first solver, divided by (|flux| + 1)).
    """
    from scipy.optimize import lsq_linear
    if "normal" in solvers:
        from .pipeline import _nnlsNormal     # imported before the clock starts
    out = {}
    for name in solvers:
        t0 = time.perf_counter()
        if name == "eigen":
            x = solveSparse(A, b).x
        elif name == "normal":
            x = _nnlsNormal(A, b)
        elif name == "scipy":
            x = lsq_linear(A, b, bounds=(0, np.inf), method="trf", lsmr_tol="auto",
                           max_iter=2000).x
        else:
            raise ValueError(f"unknown solver {name!r}")
        dt = time.perf_counter() - t0
        r = A @ x - b
        out[name] = dict(seconds=dt, x=x, cost=0.5 * float(r @ r))
    ref = out[solvers[0]]["x"]
    for name in solvers:
        out[name]["maxRelDiff"] = float(np.max(np.abs(out[name]["x"] - ref) / (np.abs(ref) + 1)))
    if verbose:
        print(f"  {A.shape[1]} sources, {A.shape[0]} pixels, {A.nnz / 1e6:.1f} M matrix entries")
        for name in solvers:
            o = out[name]
            print(f"    {name:7s} {o['seconds']:8.2f} s   chi2/2 = {o['cost']:.10e}   "
                  f"max |dx| / (|x| + 1) vs {solvers[0]}: {o['maxRelDiff']:.1e}")
    return out
