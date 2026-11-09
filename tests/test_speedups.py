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

"""The sped-up pipeline steps give the same answers as the original code."""

import unittest
from unittest import mock

import numpy as np

import lsstStubs
from lsstStubs import Box2I, Point2I

lsstStubs.install()

from lsst.meas.crowd import fitLib as L  # noqa: E402
from lsst.meas.crowd import pipeline as pl  # noqa: E402
from test_sparseSolver import _gaussStamp  # noqa: E402


def _brightStarCorrectionsRef(exposure, S, fit, model, psfPeak, noise, fwhm, badBitmask,
                              sigmaMin=100.0, radiusFwhm=4.0, sweeps=2, radial=True,
                              zones=None, haloModes=0, brightModes=0, coreRings=(),
                              coreSigma=0.6, coreModes=0):
    """The original implementation (before the basis / factorisation reuse)."""
    data, good = exposure.image.array, fit["good"]
    var = exposure.variance.array
    bb = exposure.getBBox()
    X0, Y0 = bb.getMinX(), bb.getMinY()
    amps = fit["amps"]
    sat = (exposure.mask.array & exposure.mask.getPlaneBitMask("SAT")) != 0
    corr = np.zeros_like(model)
    snr = amps * psfPeak / noise
    picks = set(np.flatnonzero(snr > sigmaMin).tolist())
    if sat.any():
        for j, (x, y) in enumerate(S["xy"]):
            ix, iy = int(round(x)) - X0, int(round(y)) - Y0
            if 0 <= iy < sat.shape[0] and 0 <= ix < sat.shape[1] and sat[iy, ix] and amps[j] > 0:
                picks.add(j)
    order = sorted(picks, key=lambda j: -amps[j])
    R = radiusFwhm * fwhm
    imgs, info = {}, {}
    for sweep in range(sweeps):
        for j in order:
            c = fit["comps"][j]
            if c is None or amps[j] <= 0:
                continue
            ys, xs = c["ys"], c["xs"]
            Pm = c["stamp"]
            gy, gx = np.gradient(Pm)
            gyy, gyx = np.gradient(gy)
            _, gxx = np.gradient(gx)
            yy, xx = np.mgrid[ys.start:ys.stop, xs.start:xs.stop]
            px, py = S["xy"][j, 0] - X0, S["xy"][j, 1] - Y0
            rr = np.hypot(xx - px, yy - py)
            Rj, rks, isSat = R, (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5), False
            if zones is not None and len(zones):           # saturated star: model its halo
                dz = np.hypot(zones[:, 0] - S["xy"][j, 0], zones[:, 1] - S["xy"][j, 1])
                k = int(np.argmin(dz))
                if dz[k] < 0.5 * zones[k, 2]:
                    Rj = max(R, zones[k, 2])
                    rks = tuple(np.arange(0.5, Rj / fwhm + 0.01, 0.75))
                    isSat = True
            okpix = good[ys, xs] & (var[ys, xs] > 0)
            terms = [Pm, gx, gy, gxx, gyy, gyx]
            if radial:                  # smooth rings: core/halo profile differs from the PSF
                for rk in rks:
                    ring = np.exp(-0.5 * ((rr - rk * fwhm) / (0.5 * fwhm)) ** 2)
                    if (okpix & (ring > 0.5)).sum() >= 10:     # skip rings inside the mask
                        terms.append(Pm.max() * ring)
                        nm = (haloModes if isSat else (brightModes if rk <= 4.0 else 0))
                        if nm:                                 # non-circular structure
                            th = np.arctan2(yy - py, xx - px)
                            for mm in range(1, int(nm) + 1):
                                terms += [Pm.max() * ring * np.cos(mm * th),
                                          Pm.max() * ring * np.sin(mm * th)]
            if coreRings and not isSat:   # narrow core structure (1-2 px scale)
                thc = np.arctan2(yy - py, xx - px)
                for rk in coreRings:
                    ring = np.exp(-0.5 * ((rr - rk) / coreSigma) ** 2)
                    if (okpix & (ring > 0.5)).sum() >= 1:
                        terms.append(Pm.max() * ring)
                        base = ring * (rr / coreSigma if rk == 0 else 1.0)  # spike: its own dipole
                        for mm in range(1, int(coreModes) + 1):
                            terms += [Pm.max() * base * np.cos(mm * thc),
                                      Pm.max() * base * np.sin(mm * thc)]
            T = np.array(terms) * amps[j]
            if j in imgs:
                corr[ys, xs] -= imgs[j]
            sel = okpix & (rr < Rj)
            if sel.sum() < 30:
                continue
            w = 1.0 / np.sqrt(var[ys, xs][sel])
            r = (data[ys, xs] - model[ys, xs] - corr[ys, xs])[sel] * w
            coef, *_ = np.linalg.lstsq(T[:, sel].T * w[:, None], r, rcond=1e-8)
            imgs[j] = np.tensordot(coef, T, axes=1)
            corr[ys, xs] += imgs[j]
            if sweep == sweeps - 1:
                r2 = r - (imgs[j][sel] * w)
                ring = np.hypot(xx - px, yy - py) < R
                info[j] = dict(x=float(S["xy"][j, 0]), y=float(S["xy"][j, 1]), snr=float(snr[j]),
                               nSat=int((sat[ys, xs] & ring).sum()), flux=float(coef[0] * amps[j]),
                               dfluxPct=float(100 * coef[0]),
                               dx=float(-coef[1]), dy=float(-coef[2]),
                               chi2Before=float(np.mean(r ** 2)), chi2After=float(np.mean(r2 ** 2)))
    return corr, info


class LstsqSolverTestCase(unittest.TestCase):
    def testMatchesLstsq(self):
        """Well-conditioned (Gram path), ill-conditioned and rank-deficient (QR + SVD path)."""
        rng = np.random.default_rng(0)
        A = rng.normal(size=(500, 30))
        bad = A.copy()
        bad[:, 5] = bad[:, 4] + 1e-7 * rng.normal(size=500)        # nearly collinear
        deficient = A.copy()
        deficient[:, 7] = 2 * deficient[:, 3]                     # exactly collinear
        zeroCol = A.copy()
        zeroCol[:, 0] = 0
        wide = A[:20]                                              # fewer rows than columns
        for M in (A, bad, deficient, zeroCol, wide, A * np.logspace(0, 6, 30)):
            solve = pl._lstsqSolver(M)
            for r in (rng.normal(size=M.shape[0]), rng.normal(size=M.shape[0])):
                ref = np.linalg.lstsq(M, r, rcond=1e-8)[0]
                np.testing.assert_allclose(solve(r), ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max())


class BrightStarCorrectionsTestCase(unittest.TestCase):
    def testSameAsOriginal(self):
        """Bright and saturated stars on a crowded tile: same correction image and table."""
        n, sigma, half = 200, 2.67, 40
        rng = np.random.default_rng(1)
        xy = np.vstack([rng.uniform(40, 160, (8, 2)), rng.uniform(0, n, (120, 2))])
        flux = np.r_[10 ** rng.uniform(5.3, 6.5, 8), 10 ** rng.uniform(2.5, 4.3, 120)]
        stamp, truthStamp = _gaussStamp(sigma, half), _gaussStamp(1.05 * sigma, half)
        box = Box2I(Point2I(0, 0), Point2I(n - 1, n - 1))
        model = np.zeros((n, n))
        for (x, y), f in zip(xy, flux):
            arr, bb = truthStamp(None, x, y, box)
            model[bb.getMinY():bb.getMaxY() + 1, bb.getMinX():bb.getMaxX() + 1] += f * arr
        var = 400 + model
        data = model + rng.normal(size=model.shape) * np.sqrt(var)
        mask = np.zeros((n, n), int)
        yy, xx = np.mgrid[:n, :n]
        mask[(xx - xy[0, 0]) ** 2 + (yy - xy[0, 1]) ** 2 < 25] = 1   # a saturated core
        exp = lsstStubs.FakeExposure(data, var, mask=mask)
        pl.enableSpeedups()
        try:
            with mock.patch.object(L, "psfStamp", stamp):
                fit = L.jointPsfFitDiag(exp, None, [tuple(p) for p in xy], 0)
        finally:
            pl.disableSpeedups()
        psfPeak = stamp(None, 100.0, 100.0, box)[0].max()
        args = (exp, dict(xy=xy), fit, fit["modelFull"], psfPeak, 20.0, 2.3548 * sigma, 1)
        kw = dict(radial=True, zones=np.array([[xy[0, 0], xy[0, 1], 20.0]]), haloModes=4,
                  brightModes=4, coreRings=(0.0, 1.0, 2.0, 3.0, 4.0), coreSigma=0.6, coreModes=2)
        corr, info = pl.brightStarCorrections(*args, **kw)
        corrRef, infoRef = _brightStarCorrectionsRef(*args, **kw)
        self.assertGreater(len(info), 5)
        self.assertEqual(sorted(info), sorted(infoRef))
        np.testing.assert_allclose(corr, corrRef, rtol=0, atol=1e-8 * np.abs(corrRef).max())
        for j in info:
            for k, v in infoRef[j].items():
                self.assertAlmostEqual(info[j][k], v, delta=1e-7 * max(1.0, abs(v)), msg=f"{j} {k}")


class WindowedDerivativesTestCase(unittest.TestCase):
    @unittest.skipUnless(__import__("lsst.meas.crowd.fastKernels", fromlist=["x"])._HAVE_WINDOW,
                         "C++ extension without renderDerivativesWindow")
    def testSameAsFullStamp(self):
        """Derivatives rendered over a small clip window equal the full stamps cropped to it."""
        import lsst.meas.crowd.fastKernels as fk
        rng = np.random.default_rng(0)
        K = rng.random((81, 81))
        f = fk.FastPsf(K / K.sum())
        nWin = 0
        for _ in range(200):
            x, y = rng.uniform(-30, 500, 2)
            h = int(rng.integers(3, 15))
            clip = (int(x) - h, int(y) - h, int(x) + h, int(y) + h)
            if rng.random() < 0.3:                         # clipped at an image edge too
                clip = (max(clip[0], 0), max(clip[1], 0), min(clip[2], 459), min(clip[3], 459))
            a = f.derivatives(x, y, clip)
            with mock.patch.object(fk, "_HAVE_WINDOW", False):
                b = f.derivatives(x, y, clip)
            if a[0] is None or b[0] is None:
                self.assertTrue(a[0] is None and b[0] is None)
                continue
            nWin += 1
            self.assertEqual(a[2], b[2])
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
        self.assertGreater(nWin, 100)


class SnrFromFitMatrixTestCase(unittest.TestCase):
    def testSameAsRebuilt(self):
        """The S/N from the fit's stored weighted matrix equals the one rebuilt from the stamps."""
        n, sigma, half = 120, 1.7, 12
        rng = np.random.default_rng(3)
        xy = rng.uniform(5, 115, (150, 2))
        stamp = _gaussStamp(sigma, half)
        box = Box2I(Point2I(0, 0), Point2I(n - 1, n - 1))
        model = np.zeros((n, n))
        for (x, y), f in zip(xy, 10 ** rng.uniform(2, 4, len(xy))):
            arr, bb = stamp(None, x, y, box)
            model[bb.getMinY():bb.getMaxY() + 1, bb.getMinX():bb.getMaxX() + 1] += f * arr
        exp = lsstStubs.FakeExposure(model + rng.normal(size=model.shape) * np.sqrt(100 + model),
                                     100 + model)
        pl.enableSpeedups()
        try:
            with mock.patch.object(L, "psfStamp", stamp):
                fit = L.jointPsfFitDiag(exp, None, [tuple(p) for p in xy], 0)
        finally:
            pl.disableSpeedups()
        self.assertIn("Aw", fit)
        fast = pl.ampSnrFullFast(fit, 1.3)
        rebuilt = pl.ampSnrFullFast({k: v for k, v in fit.items() if k != "Aw"}, 1.3)
        np.testing.assert_allclose(fast, rebuilt, rtol=1e-10, atol=1e-12)


class _KernelPsf(lsstStubs.FakePsf):
    """A constant PSF whose computeImage renders with the fast kernels (so the pipeline's fast
    path validates and switches on, as for a real visit PSF)."""

    def __init__(self, kernel):
        super().__init__(kernel)
        from lsst.meas.crowd.fastKernels import FastPsf
        self._fast = FastPsf(self.kernel)

    def computeImage(self, point):
        arr, b = self._fast.stamp(point[0], point[1], (-10 ** 8, -10 ** 8, 10 ** 8, 10 ** 8), None)
        g = L.geom                  # the pipeline's geom (lsst.geom with the stack): same box type
        bb = g.Box2I(g.Point2I(int(b[0]), int(b[1])), g.Point2I(int(b[2]), int(b[3])))
        return __import__("types").SimpleNamespace(array=arr, getBBox=lambda: g.Box2I(bb))


def _crowdedScene(n=150, nSrc=260, seed=5, half=20, sigma=1.8):
    rng = np.random.default_rng(seed)
    u = np.arange(-half, half + 1)
    K = np.exp(-0.5 * (u[None] ** 2 + u[:, None] ** 2) / sigma ** 2)
    K /= K.sum()
    psf = _KernelPsf(K)
    xy = rng.uniform(-3, n + 2, (nSrc, 2)) + 1000.0          # parent coordinates, origin 1000
    flux = 10 ** rng.uniform(2.3, 4.5, nSrc)
    model = np.zeros((n, n))
    box = Box2I(Point2I(1000, 1000), Point2I(1000 + n - 1, 1000 + n - 1))
    for (x, y), f in zip(xy, flux):
        a, bb = psf._fast.stamp(x + rng.normal(0, 0.2), y + rng.normal(0, 0.2),
                                (1000, 1000, 1000 + n - 1, 1000 + n - 1), None)
        if a is not None:
            model[bb[1] - 1000:bb[3] - 999, bb[0] - 1000:bb[2] - 999] += f * a
    var = 100 + model
    data = model + rng.normal(size=model.shape) * np.sqrt(var)
    mask = np.zeros((n, n), int)
    mask[40:44, 60:90] = 4                                     # some bad pixels
    exp = lsstStubs.FakeExposure(data, var, mask=mask, psf=psf, x0=1000, y0=1000)
    return exp, psf, xy, flux, box


@unittest.skipUnless(hasattr(pl._fkLib(), "buildDesignMatrix"), "C++ extension without the loops")
class CxxLoopsTestCase(unittest.TestCase):
    """The per-source loops moved to C++ give the Python loops' results."""

    def setUp(self):
        pl.enableSpeedups()
        pl.clearStampCache()
        self.exp, self.psf, self.xy, self.flux, self.box = _crowdedScene()

    def tearDown(self):
        pl.disableSpeedups()

    def testStampBatchSameAsOneByOne(self):
        pl.setStampSizes(self.xy, self.flux, self.psf, None, 10.0, 4.2, 0.05)   # crops some stamps
        batch = pl._psfStampBatchCached(self.psf, self.xy, self.box, taper=L.WING_TAPER)
        pl.clearStampCache()
        pl.setStampSizes(self.xy, self.flux, self.psf, None, 10.0, 4.2, 0.05)
        for (x, y), (a, b) in zip(self.xy, batch):
            ref, bb = pl._psfStampCached(self.psf, x, y, self.box, taper=L.WING_TAPER)
            if ref is None:
                self.assertIsNone(a)
                continue
            self.assertEqual(b, (bb.getMinX(), bb.getMinY(), bb.getMaxX(), bb.getMaxY()))
            np.testing.assert_allclose(a, ref, rtol=0, atol=1e-14 * np.abs(ref).max())

    def testDesignMatrixAndModel(self):
        pl._SOLVER["last"] = None                                # same (cold) start for both solves
        fit = L.jointPsfFitDiag(self.exp, self.psf, [tuple(p) for p in self.xy], 4)
        pl._SOLVER["last"] = None
        with mock.patch.object(L, "_lib", None):                 # the Python loops
            ref = L.jointPsfFitDiag(self.exp, self.psf, [tuple(p) for p in self.xy], 4)
        self.assertEqual((fit["Aw"] != ref["Aw"]).nnz, 0)        # identical matrix ...
        np.testing.assert_array_equal(fit["groupLabels"], ref["groupLabels"])
        self.assertEqual(fit["nGroups"], ref["nGroups"])
        np.testing.assert_array_equal(fit["groupMap"], ref["groupMap"])
        np.testing.assert_array_equal(fit["amps"], ref["amps"])   # ... hence identical fluxes
        np.testing.assert_array_equal(fit["modelFull"], ref["modelFull"])
        for c, r in zip(fit["comps"], ref["comps"]):
            self.assertEqual(c is None, r is None)
            if c is not None:
                np.testing.assert_array_equal(c["sig"], r["sig"])
                self.assertEqual((c["ys"], c["xs"], c["bboxParent"]), (r["ys"], r["xs"], r["bboxParent"]))
        for g, v in ref["groupChi2"].items():
            self.assertTrue((np.isnan(v) and np.isnan(fit["groupChi2"][g]))
                            or abs(v - fit["groupChi2"][g]) <= 1e-12 * max(1.0, abs(v)))

    def testCentroidStepSameAsPython(self):
        fit = L.jointPsfFitDiag(self.exp, self.psf, [tuple(p) for p in self.xy], 4)
        S = pl._state(self.xy)
        S["ext"][:3] = 1.0                                         # extended: stay put
        idx = np.arange(len(self.xy))
        args = (self.exp, self.psf, S, fit, idx, 0.8, 50.0, 4.0, 1.8)
        new = pl._refineSubset(*args)
        ref = pl._refineSubsetPython(*args)
        moved = np.any(ref != self.xy, axis=1)
        self.assertGreater(moved.sum(), 50)
        np.testing.assert_array_equal(np.any(new != self.xy, axis=1), moved)
        np.testing.assert_allclose(new, ref, rtol=0, atol=1e-9)


class SkyTermsTestCase(unittest.TestCase):
    """The joint fit's sky terms (skyTerms=3) recover a background plane with the fluxes."""

    def setUp(self):
        self.n, sigma, half = 160, 1.8, 12
        rng = np.random.default_rng(8)
        self.xy = rng.uniform(10, 150, (120, 2))
        self.flux = 10 ** rng.uniform(3.3, 4.5, len(self.xy))
        self.stamp = _gaussStamp(sigma, half)
        box = Box2I(Point2I(0, 0), Point2I(self.n - 1, self.n - 1))
        model = np.zeros((self.n, self.n))
        for (x, y), f in zip(self.xy, self.flux):
            a, bb = self.stamp(None, x, y, box)
            model[bb.getMinY():bb.getMaxY() + 1, bb.getMinX():bb.getMaxX() + 1] += f * a
        yy, xx = np.mgrid[:self.n, :self.n]
        self.sky = -9.0 + 4.0 * (xx - self.n / 2) / self.n - 2.0 * (yy - self.n / 2) / self.n
        var = 100 + model
        self.data = model + self.sky + rng.normal(size=model.shape) * np.sqrt(var)
        self.var = var

    def _fit(self, solver):
        exp = lsstStubs.FakeExposure(self.data.copy(), self.var.copy())
        pl.enableSpeedups()
        pl.setJointSolver(solver)
        pl._SOLVER["last"] = None
        try:
            with mock.patch.object(L, "psfStamp", self.stamp):
                return exp, L.jointPsfFitDiag(exp, None, [tuple(p) for p in self.xy], 0, skyTerms=3)
        finally:
            pl.setJointSolver("eigen")
            pl.disableSpeedups()

    def testRecoversPlaneAndFluxes(self):
        exp, fit = self._fit("eigen")
        np.testing.assert_allclose(fit["sky"], [-9.0, 4.0, -2.0], atol=0.3)
        np.testing.assert_allclose(fit["skyImage"], self.sky, atol=0.4)
        ratio = fit["amps"] / self.flux                            # no flux bias from the sky
        self.assertLess(abs(np.median(ratio) - 1), 0.01)
        self.assertLess(np.percentile(np.abs(ratio - 1), 90), 0.06)
        _, ref = self._fit("scipy")                                # scipy's bounded solver
        np.testing.assert_allclose(fit["sky"], ref["sky"], atol=1e-3)
        np.testing.assert_allclose(fit["amps"], ref["amps"], rtol=1e-4, atol=1e-2)

    def testFitSubtractsSky(self):
        """`pipeline._fit` leaves background-subtracted data and a sources-only model."""
        exp = lsstStubs.FakeExposure(self.data.copy(), self.var.copy())
        S = pl._state(self.xy)
        pl.enableSpeedups()
        pl._SOLVER["last"] = None
        try:
            with mock.patch.object(L, "psfStamp", self.stamp):
                fit = pl._fit(exp, None, S, 0, None, dict(skyTerms=3))
        finally:
            pl.disableSpeedups()
        np.testing.assert_allclose(exp.image.array, self.data - fit["skyImage"], atol=1e-4)
        resid = exp.image.array - fit["modelFull"]
        self.assertLess(abs(np.median(resid)), 1.0)


class MedianSkyTestCase(unittest.TestCase):
    """`pipeline._medianSkyPlane`: a sky plane robust to faint unmodelled stars (skyMedian)."""

    def testRecoversPlaneDespitePositiveTail(self):
        n = 256
        rng = np.random.default_rng(11)
        yy, xx = np.mgrid[:n, :n]
        plane = -4.0 + 3.0 * (xx - n / 2) / n - 1.5 * (yy - n / 2) / n
        tail = rng.exponential(25.0, (n, n)) * (rng.uniform(size=(n, n)) < 0.15)   # faint stars
        resid = plane + rng.normal(0, 20, (n, n)) + tail
        ok = np.ones((n, n), bool)
        ok[:, :40] = False                                        # a masked strip
        fit, coef, nb = pl._medianSkyPlane(resid, ok)
        np.testing.assert_allclose(coef[1:], [3.0, -1.5], atol=0.8)          # the tilt
        meanBias = resid[ok].mean() - plane[ok].mean()                       # ~ +3.75: the tail
        self.assertGreater(meanBias, 3.0)
        self.assertLess(abs(coef[0] - (-4.0)), 0.7 * meanBias)               # the median: far less
        self.assertEqual(nb, 8 * 7)                               # the masked strip's blocks are skipped
        np.testing.assert_allclose(fit, coef[0] + coef[1] * (xx - n / 2) / n + coef[2] * (yy - n / 2) / n)

    def testTooFewBlocks(self):
        fit, coef, nb = pl._medianSkyPlane(np.ones((40, 40)), np.zeros((40, 40), bool))
        self.assertEqual(nb, 0)
        np.testing.assert_array_equal(coef, 0)
        np.testing.assert_array_equal(fit, 0)

    def testFastPreset(self):
        from lsst.meas.crowd import session
        P = pl._params(session.makeConfig(session.FAST_STELLAR_FIELD_CONFIG))
        self.assertTrue(P["skyMedian"])
        self.assertEqual(P["skyTerms"], 0)


if __name__ == "__main__":
    unittest.main()
