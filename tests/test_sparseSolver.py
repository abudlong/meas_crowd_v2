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

"""Tests of the sparse joint-fit solver (pipe_crowd's Eigen LSCG + non-negativity active set)."""

import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear, nnls

import lsstStubs
from lsstStubs import Box2I, Point2I   # the box type FakeExposure uses, with or without the stack

lsstStubs.install()

from lsst.meas.crowd import fitLib as L  # noqa: E402
from lsst.meas.crowd import pipeline as pl  # noqa: E402
from lsst.meas.crowd import sparseSolver as ss  # noqa: E402

needsSolver = unittest.skipUnless(ss.HAVE_SPARSE_SOLVER, "C++ sparse solver not built")


@needsSolver
class SolveSparseTestCase(unittest.TestCase):
    def testMatchesNnls(self):
        """Non-negative solve equals scipy's exact NNLS, also when most values hit the bound."""
        rng = np.random.default_rng(1)
        for seed in range(5):
            A = sp.random(400, 60, density=0.2, random_state=seed, format="csc")
            b = rng.normal(size=400)
            r = ss.solveSparse(A, b)
            xRef, _ = nnls(A.toarray(), b)
            self.assertTrue(r.converged)
            self.assertEqual(r.nAtBound, int(np.sum(xRef == 0)))
            np.testing.assert_allclose(r.x, xRef, atol=1e-7)

    def testUnconstrainedMatchesLstsq(self):
        """nonNegative=False is pipe_crowd's plain LSCG solve: equals the least-squares answer."""
        rng = np.random.default_rng(2)
        A = sp.random(300, 40, density=0.3, random_state=3, format="csr")
        b = rng.normal(size=300)
        r = ss.solveSparse(A, b, nonNegative=False)
        xRef = np.linalg.lstsq(A.toarray(), b, rcond=None)[0]
        self.assertTrue(r.converged)
        self.assertEqual(r.activeSetIterations, 1)
        self.assertTrue((r.x < 0).any())
        np.testing.assert_allclose(r.x, xRef, atol=1e-7)

    def testEmptyColumnAndWarmStart(self):
        """A column without pixels stays 0; a warm start gives the same answer."""
        rng = np.random.default_rng(4)
        A = sp.random(200, 20, density=0.3, random_state=5, format="lil")
        A[:, 7] = 0
        A = A.tocsc()
        b = A @ rng.uniform(1, 2, 20) + 0.01 * rng.normal(size=200)
        cold = ss.solveSparse(A, b)
        warm = ss.solveSparse(A, b, x0=cold.x * 1.1)
        self.assertEqual(cold.x[7], 0.0)
        np.testing.assert_allclose(warm.x, cold.x, atol=1e-7)

    def testCrowdedField(self):
        """On a synthetic crowded tile the fluxes match scipy's lsq_linear and the truth."""
        A, b, truth = ss.makeCrowdedFieldProblem(nSources=250, size=200, seed=3)
        r = ss.solveSparse(A, b)
        ref = lsq_linear(A, b, bounds=(0, np.inf), method="trf", lsmr_tol="auto", max_iter=2000)
        cost = 0.5 * np.sum((A @ r.x - b) ** 2)
        self.assertTrue(r.converged)
        self.assertLessEqual(cost, ref.cost * (1 + 1e-9))
        bright = truth > 20 * np.median(truth)
        np.testing.assert_allclose(r.x[bright], truth[bright], rtol=0.05)

    def testUnboundedColumns(self):
        """Columns marked unbounded may go negative; the rest stay >= 0 (scipy's exact BVLS)."""
        rng = np.random.default_rng(6)
        A = sp.random(500, 40, density=0.2, random_state=1, format="csc")
        b = rng.normal(size=500) - 0.5
        free = np.zeros(40, bool)
        free[-3:] = True
        r = ss.solveSparse(A, b, unbounded=free)
        ref = lsq_linear(A.toarray(), b, bounds=(np.where(free, -np.inf, 0.0), np.inf),
                         method="bvls").x
        self.assertTrue(r.converged)
        self.assertTrue((r.x[free] < 0).all())
        self.assertGreaterEqual(r.x[~free].min(), 0.0)
        np.testing.assert_allclose(r.x, ref, atol=1e-8)

    def testBadInput(self):
        A = sp.random(20, 5, density=0.5, random_state=0, format="csc")
        with self.assertRaises(ValueError):
            ss.solveSparse(A, np.ones(19))
        with self.assertRaises(ValueError):
            ss.solveSparse(A, np.ones(20), x0=np.ones(4))


def _gaussStamp(sigma, half):
    def stamp(psf, x, y, clipBbox, taper=None):
        ix, iy = int(round(x)), int(round(y))
        yy, xx = np.mgrid[iy - half:iy + half + 1, ix - half:ix + half + 1]
        arr = np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sigma ** 2) / (2 * np.pi * sigma ** 2)
        full = Box2I(Point2I(ix - half, iy - half), Point2I(ix + half, iy + half))
        bb = Box2I(full)
        bb.clip(clipBbox)
        if bb.isEmpty():
            return None, bb
        return arr[bb.getMinY() - full.getMinY():bb.getMaxY() - full.getMinY() + 1,
                   bb.getMinX() - full.getMinX():bb.getMaxX() - full.getMinX() + 1], bb
    return stamp


class PipelineSolverTestCase(unittest.TestCase):
    def setUp(self):
        self.saved = dict(pl._SOLVER)

    def tearDown(self):
        pl._SOLVER.update(self.saved)

    def testOption(self):
        """jointSolver is a pipeline option, "eigen" by default, and is validated."""
        self.assertEqual(pl.PIPELINE_DEFAULTS["jointSolver"], "eigen")
        self.assertEqual(pl._params({"jointSolver": "normal"})["jointSolver"], "normal")
        with self.assertRaises(ValueError):
            pl.setJointSolver("lapack")

    @needsSolver
    def testJointFitSameWithEverySolver(self):
        """`fitLib.jointPsfFitDiag` gives the same fit with "eigen", "normal" and "scipy"."""
        n, sigma, half = 120, 1.7, 12
        rng = np.random.default_rng(0)
        xy = rng.uniform(5, 115, (350, 2))      # >= 300 sources: "normal" uses its NNLS path
        flux = 10 ** rng.uniform(2, 4, len(xy))
        stamp = _gaussStamp(sigma, half)
        model = np.zeros((n, n))
        for (x, y), f in zip(xy, flux):
            arr, bb = stamp(None, x, y, Box2I(Point2I(0, 0), Point2I(n - 1, n - 1)))
            model[bb.getMinY():bb.getMaxY() + 1, bb.getMinX():bb.getMaxX() + 1] += f * arr
        data = model + rng.normal(size=model.shape) * np.sqrt(100 + model)
        exp = lsstStubs.FakeExposure(data, 100 + model)
        pl.enableSpeedups()
        fits = {}
        try:
            with mock.patch.object(L, "psfStamp", stamp):
                for name in pl.JOINT_SOLVERS:
                    pl.setJointSolver(name)
                    pl.solverStats(reset=True, verbose=False)
                    fits[name] = L.jointPsfFitDiag(exp, None, [tuple(p) for p in xy], 0)
                    st = pl.solverStats(verbose=False)
                    self.assertEqual(st["calls"], 1)
                    self.assertEqual(st["fallbacks"], 0)
        finally:
            pl.disableSpeedups()
        ref = fits["scipy"]
        for name in ("eigen", "normal"):
            self.assertLessEqual(fits[name]["lsqCost"], ref["lsqCost"] * (1 + 1e-7))
            np.testing.assert_allclose(fits[name]["amps"], ref["amps"], atol=1e-3 * flux.max())
            np.testing.assert_allclose(fits[name]["modelFull"], ref["modelFull"],
                                       atol=1e-3 * model.max())


if __name__ == "__main__":
    unittest.main()
