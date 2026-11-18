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

"""Tests of the whole-detector tiling, per-tile products, stitching and plotting.  The per-tile
fit itself is replaced by a perfect synthetic model (it needs the stack); everything around it
-- task splitting, resuming, files, ownership, mosaic -- is the real code."""

import contextlib
import io
import os
import tempfile
import types
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import lsstStubs  # noqa: E402

lsstStubs.install()

from lsst.meas.crowd import detector as det  # noqa: E402
from lsst.meas.crowd import gaiaCatalog, pipeline, session  # noqa: E402

geom = pipeline.L.geom
plt.show = lambda *a, **k: None


class TileSpecTestCase(unittest.TestCase):
    def testPartition(self):
        for (w, h, t) in ((4072, 4000, 400), (600, 500, 200), (100, 90, 400), (1001, 999, 250)):
            specs = det.tileSpecs(w, h, t, x0=7, y0=3)
            cover = np.zeros((h, w), int)
            for x0, y0, tw, th in specs:
                self.assertLessEqual(tw, t + 1)
                cover[y0 - 3:y0 - 3 + th, x0 - 7:x0 - 7 + tw] += 1
            self.assertTrue(np.all(cover == 1), (w, h, t))      # every pixel in exactly one tile
        self.assertEqual(len(det.tileSpecs(4072, 4000, 400)), 11 * 10)

    def testRegion(self):
        specs = det.tileSpecs(600, 500, 200, region=(250, 330, 0, 50))
        self.assertEqual([s[:2] for s in specs], [(200, 0)])
        self.assertEqual(det.tileName((200, 0, 200, 167)), "x0200_y0000_200x167")


class _Synthetic:
    """A 'detector' with Gaussian stars and a perfect per-tile fit."""

    def __init__(self, shape=(500, 600), n=60, sigma=2.0, half=12, seed=1):
        rng = np.random.default_rng(seed)
        self.shape, self.sigma, self.half = shape, sigma, half
        self.xy = np.column_stack([rng.uniform(5, shape[1] - 5, n), rng.uniform(5, shape[0] - 5, n)])
        self.flux = 10 ** rng.uniform(3.5, 5, n)
        truth = np.zeros(shape)
        for (x, y), f in zip(self.xy, self.flux):
            truth += lsstStubs.place(shape, lsstStubs.gaussianComp(shape, x, y, sigma, half), f)
        self.truth = truth
        self.var = 100 + truth
        self.data = truth + rng.normal(0, 1, shape) * np.sqrt(self.var)
        self.psf = lsstStubs.FakePsf(lsstStubs.gaussianKernel(sigma, half))
        self.calls, self.kws, self.psfs = [], [], []
        self.chi2For = {}
        bb = geom.Box2I(geom.Point2I(0, 0), geom.Point2I(shape[1] - 1, shape[0] - 1))
        self.visitImage = types.SimpleNamespace(getBBox=lambda: bb)

    def prepareSession(self, visit, detector, cutouts, pad=0, psfSource=None, visitImage=None):
        jobs = {}
        for c in cutouts:
            ib = geom.Box2I(geom.Point2I(c["x0"], c["y0"]),
                            geom.Point2I(c["x0"] + c["w"] - 1, c["y0"] + c["h"] - 1))
            fx0, fy0 = max(c["x0"] - pad, 0), max(c["y0"] - pad, 0)
            fx1 = min(c["x0"] + c["w"] - 1 + pad, self.shape[1] - 1)
            fy1 = min(c["y0"] + c["h"] - 1 + pad, self.shape[0] - 1)
            sl = (slice(fy0, fy1 + 1), slice(fx0, fx1 + 1))
            exp = lsstStubs.FakeExposure(self.data[sl].copy(), self.var[sl].copy(), psf=self.psf,
                                         x0=fx0, y0=fy0)
            jobs[c["name"]] = dict(interior=ib, raw=exp)
        return types.SimpleNamespace(jobs=jobs, gaia=None)

    def runSessionGaia(self, sess, cfg, names=None, plotFinal=False, **kw):
        name = names[0]
        self.calls.append(name)
        job = sess.jobs[name]
        exp = job["raw"]
        bb = exp.getBBox()
        x0, y0 = bb.getMinX(), bb.getMinY()
        shape = exp.image.array.shape
        inside = ((self.xy[:, 0] > x0 - 3) & (self.xy[:, 0] < bb.getMaxX() + 3) &
                  (self.xy[:, 1] > y0 - 3) & (self.xy[:, 1] < bb.getMaxY() + 3))
        xy, fl = self.xy[inside], self.flux[inside]
        comps = [lsstStubs.gaussianComp(shape, x, y, self.sigma, self.half, x0, y0) for x, y in xy]
        model = self.truth[y0:y0 + shape[0], x0:x0 + shape[1]].copy()
        S = pipeline._state(xy, fromGaia=True)
        fit = dict(amps=fl, modelFull=model, good=np.ones(shape, bool), variance=exp.variance.array,
                   comps=comps)
        res = dict(S=S, snaps=[dict(fit=fit)], modelFull=model, amps=fl.copy(), brightCorr={},
                   peaks=[tuple(p) for p in xy])
        self.kws.append(kw)
        self.psfs.append(getattr(sess, "psf", None))
        # the pipeline's PSF steps: with refitPsf="piff" and two passes allowed, the first pass is
        # checked (psfRefitMaxChi2 > 0) and refitted when it fails or the check is off
        h1 = dict(exposure=exp, result=res, psf=self.psf)
        hist = [h1]
        thr = kw.get("psfRefitMaxChi2", 0.0)
        chi2 = self.chi2For.get(name, 1.0)
        if thr:
            h1["psfChi2"] = chi2
        if kw.get("refitPsf") == "piff" and kw.get("maxOuter", 3) >= 2 and (not thr or chi2 > thr):
            h1["psfRefitRaw"] = lsstStubs.FakePsf(self.localKernel(x0))
            hist.append(dict(exposure=exp, result=res, psf=self.psf))
        run = types.SimpleNamespace(name=name, outerHistory=hist,
                                    badBitmask=4, interiorBbox=job["interior"], config={})
        return {name: run}

    def localKernel(self, x0):
        """The 'local PIFF PSF' of a tile: wider to the right, so the interpolation shows."""
        return lsstStubs.gaussianKernel(1.5 + x0 / 400.0, 6)


class _FullImage(lsstStubs.FakeExposure):
    """A visit image that can be copied (for the detector PSF's clean image)."""

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def setPsf(self, p):
        self._psf = p


class SampleTilesTestCase(unittest.TestCase):
    def test(self):
        specs = det.tileSpecs(4096, 4004, 400)
        picks = det.sampleTiles(specs, 16, 4096, 4004)
        self.assertEqual(len(picks), 16)
        self.assertEqual(len(set(picks)), 16)
        cx = np.array([s[0] + s[2] / 2 for s in picks])
        cy = np.array([s[1] + s[3] / 2 for s in picks])
        for c, n in ((cx, 4096), (cy, 4004)):
            self.assertEqual(len(np.unique(c)), 4)  # 4 columns and rows, spread out
            self.assertLess(c.min(), 0.25 * n)
            self.assertGreater(c.max(), 0.75 * n)
        few = specs[:5]
        self.assertEqual(det.sampleTiles(few, 16, 4096, 4004), few)
        names = {det.tileName(s) for s in det.sampleTiles(specs, 25, 4096, 4004)}
        for corner in ("x0000_y0000", "x3724_y0000", "x0000_y3640", "x3724_y3640"):
            self.assertIn(corner, {nm[:11] for nm in names})     # the edges are sampled


class SkyExtensionTestCase(unittest.TestCase):
    """The neighbouring detectors' sky border: `buildSkyExtension` and `skyExtension`."""

    @staticmethod
    def _strip(x0, x1, y0, y1, value=-20.0, seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        v = value + rng.normal(0, 5, xx.shape)
        return dict(x=xx.ravel().astype(float), y=yy.ravel().astype(float), value=v.ravel(),
                    var=np.full(xx.size, 25.0), ok=np.ones(xx.size, bool))

    def testBorderOnlyOutsideAndWithinReach(self):
        left = self._strip(-60, -5, 0, 100)                     # a neighbour 5 px beyond x = 0
        left["value"][0] = 1e6                                  # a star core: dropped
        inside = self._strip(0, 50, 0, 100)                     # overlapping pixels: ignored
        ext = det.buildSkyExtension((0, 199, 0, 99), [left, inside], 20)
        self.assertEqual(ext["img"].shape, (140, 240))
        self.assertEqual((ext["x0"], ext["y0"], ext["e"]), (-20, -20, 20))
        cols = np.flatnonzero(ext["ok"].any(axis=0)) + ext["x0"]
        np.testing.assert_array_equal(cols, np.arange(-20, -5))   # -20 .. -6: the reach, no gap
        self.assertFalse(ext["ok"][20:120, 20:220].any())          # nothing over the detector
        self.assertEqual(ext["nPix"], 15 * 100 - 1)
        self.assertLess(abs(np.nanmedian(ext["img"]) + 20.0), 0.5)
        empty = det.buildSkyExtension((0, 199, 0, 99), [], 20)
        self.assertEqual(empty["nPix"], 0)

    def testDuplicatesAveraged(self):
        a = self._strip(-10, -5, 0, 10, value=-10.0)
        b = self._strip(-10, -5, 0, 10, value=-30.0)
        ext = det.buildSkyExtension((0, 99, 0, 99), [a, b], 20)
        self.assertLess(abs(np.nanmean(ext["img"]) + 20.0), 1.0)

    def testOffsetsReported(self):
        img = np.full((100, 200), -20.0)
        vis = lsstStubs.FakeExposure(img, np.full(img.shape, 25.0))
        strips = [self._strip(-30, 0, 0, 100, value=-26.0)]     # the neighbour's sky 6 lower
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ext = det.skyExtension(vis, 1, 2, 20, None, None, None, None, strips=strips)
        self.assertAlmostEqual(ext["offsets"]["left"], -6.0, delta=0.5)
        self.assertNotIn("right", ext["offsets"])
        self.assertIn("left -6", out.getvalue())


class ExtendSkyInputsTestCase(unittest.TestCase):
    """`pipeline._extendSkyInputs`: an edge tile's sky inputs grown by the border."""

    def setUp(self):
        strip = SkyExtensionTestCase._strip(-30, 0, 0, 300, value=-20.0)
        pipeline._SKY_EXT.clear()
        pipeline._SKY_EXT.update(det.buildSkyExtension((0, 399, 0, 299), [strip], 20))

    def tearDown(self):
        pipeline._SKY_EXT.clear()

    def test(self):
        resid = np.full((120, 100), -20.0)
        quiet = np.ones(resid.shape, bool)
        sig = np.full(resid.shape, 5.0)
        inner = geom.Box2I(geom.Point2I(150, 100), geom.Point2I(249, 219))
        self.assertIsNone(pipeline._extendSkyInputs(inner, resid, quiet, sig))   # not on an edge
        edge = geom.Box2I(geom.Point2I(0, 100), geom.Point2I(99, 219))
        rE, qE, sE, sl, n = pipeline._extendSkyInputs(edge, resid, quiet, sig)
        self.assertEqual(rE.shape, (120, 120))                  # grown 20 px to the left only
        np.testing.assert_array_equal(rE[sl], resid)
        self.assertGreater(n, 20 * 120 - 20)                    # minus the 3-sigma star cut
        self.assertGreater(qE[:, :20].mean(), 0.99)
        # the background already subtracted from the pass (a plane) is extended to the border
        yy, xx = np.mgrid[0:120, 0:100]
        bg = 3.0 + 0.1 * xx - 0.05 * yy
        rB, _, _, _, _ = pipeline._extendSkyInputs(edge, resid, quiet, sig, bgSub=bg)
        yE, xE = np.mgrid[0:120, -20:0]
        self.assertLess(abs(np.nanmedian(rB[:, :20] - (rE[:, :20] - (3.0 + 0.1 * xE
                                                                     - 0.05 * yE)))), 1e-6)


class MeshCentroidsTestCase(unittest.TestCase):
    def testPartlyCoveredCellsOnAPlane(self):
        """With ``centroids`` a plane comes out exact although some cells are half covered."""
        from lsst.meas.crowd import pipeline as pl
        yy, xx = np.mgrid[0:128, 0:160]
        plane = -20.0 + 0.2 * xx - 0.1 * yy
        ok = np.ones(plane.shape, bool)
        ok[:, :16] = False                                       # the first column of cells: half
        s0 = pl._medianSkyMesh(plane, ok, block=32, smooth=1)[0]
        s1 = pl._medianSkyMesh(plane, ok, block=32, smooth=1, centroids=True)[0]
        self.assertGreater(np.abs(s0 - plane).max(), 1.0)
        self.assertLess(np.abs(s1 - plane).max(), 0.05)


class GridPsfsTestCase(unittest.TestCase):
    def test(self):
        k1, k2 = lsstStubs.gaussianKernel(1.5, 6), lsstStubs.gaussianKernel(3.0, 8)   # sizes differ
        box = lambda x, y: geom.Box2I(geom.Point2I(x - 10, y - 10), geom.Point2I(x + 10, y + 10))  # noqa: E731
        with mock.patch.object(pipeline.L, "kernelToLsstPsf", lambda K: lsstStubs.FakePsf(K),
                               create=True):
            out = det.gridPsfs([(100.0, 100.0, k1), (300.0, 100.0, k2)],
                               dict(a=box(100, 100), mid=box(200, 100), b=box(250, 100)), 2)
        ka, kmid, kb = (out[n].kernel for n in ("a", "mid", "b"))
        self.assertEqual(ka.shape, (13, 13))                         # the common size
        np.testing.assert_allclose(ka, k1 / k1.sum(), atol=1e-12)    # on a sample: that one
        k2c = k2[2:15, 2:15] / k2[2:15, 2:15].sum()
        np.testing.assert_allclose(kmid, 0.5 * (k1 / k1.sum() + k2c), atol=1e-12)   # midway
        self.assertAlmostEqual(kb.sum(), 1.0)
        self.assertLess(kb[6, 6], kmid[6, 6])                       # nearer the wide one


class DetectorPsfTestCase(unittest.TestCase):
    """fitTiles(detectorPsf=True): a grid of local PIFF PSFs from sampled tiles."""

    def setUp(self):
        self.syn = _Synthetic()
        self.tmp = tempfile.TemporaryDirectory()

        def prepare(*a, **k):                  # the session carries the visit PSF
            sess = self.syn.prepareSession(*a, **k)
            sess.psf = self.syn.psf
            return sess

        self.patches = [mock.patch.object(session, "prepareSession", prepare),
                        mock.patch.object(gaiaCatalog, "attachGaia", lambda *a, **k: None),
                        mock.patch.object(pipeline, "runSessionGaia", self.syn.runSessionGaia),
                        mock.patch.object(pipeline.L, "kernelToLsstPsf",
                                          lambda K: lsstStubs.FakePsf(K), create=True)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _fit(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            det.fitTiles(1, 2, config=dict(detectorPsf=True, detectorPsfTiles=4,
                                           detectorPsfNeighbours=4, detectorPsfMaxChi2=5.0),
                         outDir=self.tmp.name, tile=200, pad=20, visitImage=self.syn.visitImage)
        return out.getvalue()

    def testSampledTilesFirstThenTheGrid(self):
        log = self._fit()
        picks = [det.tileName(s) for s in det.sampleTiles(det.tileSpecs(600, 500, 200), 4, 600,
                                                          500)]
        # the sampled tiles first, with the visit PSF and an unconditional refit
        self.assertEqual(sorted(self.syn.calls[:4]), sorted(picks))
        for kw, p in zip(self.syn.kws[:4], self.syn.psfs[:4]):
            self.assertEqual((kw["refitPsf"], kw["psfRefitMaxChi2"]), ("piff", 0.0))
            self.assertIs(p, self.syn.psf)
        self.assertEqual(len(self.syn.calls), 9)                     # every tile once
        # the others: from their grid PSF, checked
        for name, kw, p in zip(self.syn.calls[4:], self.syn.kws[4:], self.syn.psfs[4:]):
            self.assertEqual(kw["psfRefitMaxChi2"], 5.0)
            self.assertIsNot(p, self.syn.psf)
            self.assertAlmostEqual(p.kernel.sum(), 1.0)
        # a tile in the middle column gets a PSF between the left and right samples' widths
        mid = self.syn.calls.index("x0200_y0167_200x166")
        k = self.syn.psfs[mid].kernel
        lo = lsstStubs.gaussianKernel(1.5, 6)
        hi = lsstStubs.gaussianKernel(1.5 + 400 / 400.0, 6)
        self.assertLess(k[6, 6], lo[6, 6] / lo.sum())
        self.assertGreater(k[6, 6], hi[6, 6] / hi.sum())
        self.assertIn("4 local PIFF PSF(s)", log)
        self.assertIn("5 of the 5 other tile(s) kept the interpolated PSF, 0 refitted", log)

    def testATileWhereTheGridPsfFailsRefits(self):
        self.syn.chi2For["x0200_y0167_200x166"] = 30.0
        log = self._fit()
        self.assertIn("4 of the 5 other tile(s) kept the interpolated PSF, 1 refitted", log)
        self.assertIn("x0200_y0167_200x166", log)
        self.assertIn("PSF chi2 30.0 ->", log)


class FitStitchTestCase(unittest.TestCase):
    def setUp(self):
        self.syn = _Synthetic()
        self.tmp = tempfile.TemporaryDirectory()
        self.out = self.tmp.name
        self.patches = [mock.patch.object(session, "prepareSession", self.syn.prepareSession),
                        mock.patch.object(session, "makeConfig", lambda *a, **k: {}),
                        mock.patch.object(gaiaCatalog, "attachGaia", lambda *a, **k: None),
                        mock.patch.object(pipeline, "runSessionGaia", self.syn.runSessionGaia)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()
        plt.close("all")

    def _fit(self, **kw):
        return det.fitTiles(1, 2, outDir=self.out, tile=200, pad=20,
                            visitImage=self.syn.visitImage, **kw)

    def testTasksResumeAndStitch(self):
        a = self._fit(taskIndex=0, nTasks=2)
        b = self._fit(taskIndex=1, nTasks=2)
        self.assertEqual(len(a) + len(b), 9)                        # 3 x 3 tiles, split in two
        self.assertEqual(len(set(a) & set(b)), 0)
        nCalls = len(self.syn.calls)
        self._fit()                                                 # everything done: no refits
        self.assertEqual(len(self.syn.calls), nCalls)
        m = det.stitchTiles(self.out, verbose=False)
        np.testing.assert_allclose(m["data"], self.syn.data.astype(np.float32), rtol=1e-6)
        np.testing.assert_allclose(m["model"], self.syn.truth.astype(np.float32), rtol=1e-5, atol=1e-3)
        self.assertTrue(np.all(m["tile"] >= 0))
        z = m["resid"] / np.sqrt(m["var"])
        self.assertLess(abs(np.std(z) - 1), 0.05)
        # every star exactly once, with the right flux
        cat = m["catalog"]
        self.assertEqual(len(cat), len(self.syn.xy))
        order = np.lexsort((cat["y"], cat["x"]))
        torder = np.lexsort((self.syn.xy[:, 1], self.syn.xy[:, 0]))
        np.testing.assert_allclose(np.asarray(cat["x"])[order], self.syn.xy[torder, 0])
        np.testing.assert_allclose(np.asarray(cat["psfFlux"])[order], self.syn.flux[torder])
        # files on disk
        self.assertTrue(os.path.exists(os.path.join(self.out, "detector_mosaic.fits")))
        m2 = det.loadMosaic(self.out)
        np.testing.assert_array_equal(m2["tile"], m["tile"])
        self.assertEqual(len(m2["catalog"]), len(cat))

    def testRegionAndGaps(self):
        files = self._fit(region=(0, 150, 0, 150))                  # only the corner tile
        self.assertEqual(len(files), 1)
        m = det.stitchTiles(self.out, verbose=False)
        self.assertEqual(m["data"].shape, (167, 200))

    def testFailingTileDoesNotStopTask(self):
        good = self.syn.runSessionGaia

        def flaky(sess, cfg, names=None, **kw):
            if names[0].startswith("x0200_y0167"):
                raise RuntimeError("boom")
            return good(sess, cfg, names=names, **kw)

        with mock.patch.object(pipeline, "runSessionGaia", flaky):
            files = self._fit()
        self.assertEqual(len(files), 8)
        m = det.stitchTiles(self.out, verbose=False)
        self.assertTrue(np.any(m["tile"] < 0))                      # the gap is marked

    def testParallelWorkersSameAsSerial(self):
        """nJobs > 1 fits every tile in worker processes: same mosaic, logs, failures reported."""
        good = self.syn.runSessionGaia

        def flaky(sess, cfg, names=None, **kw):
            if names[0].startswith("x0200_y0167"):
                raise RuntimeError("boom")
            return good(sess, cfg, names=names, **kw)

        with mock.patch.object(pipeline, "runSessionGaia", flaky):
            files = self._fit(nJobs=3)
        self.assertEqual(len(files), 8)                              # the failing tile is missing
        logs = os.listdir(os.path.join(self.out, "logs"))
        self.assertEqual(len(logs), 9)
        files = self._fit(nJobs=3)                                   # rerun: only that tile
        self.assertEqual(len(files), 9)
        par = det.stitchTiles(self.out, verbose=False)
        serialOut = os.path.join(self.out, "serial")
        det.fitTiles(1, 2, outDir=serialOut, tile=200, pad=20, visitImage=self.syn.visitImage)
        ser = det.stitchTiles(serialOut, verbose=False)
        np.testing.assert_array_equal(par["model"], ser["model"])
        np.testing.assert_array_equal(par["tile"], ser["tile"])
        self.assertEqual(len(par["catalog"]), len(ser["catalog"]))

    def testConfigSettingsNotOverridden(self):
        """fitTiles' maxOuter / maxInner defaults apply only where the configuration is silent."""
        seen = []
        good = self.syn.runSessionGaia

        def spy(sess, cfg, names=None, **kw):
            seen.append(kw)
            return good(sess, cfg, names=names, **kw)

        with mock.patch.object(pipeline, "runSessionGaia", spy):
            det.fitTiles(1, 2, config=dict(maxOuter=1), outDir=self.out, tile=200, pad=20,
                         visitImage=self.syn.visitImage, region=(0, 150, 0, 150))
            self.assertNotIn("maxOuter", seen[-1])              # the configuration's 1 is used
            self.assertEqual(seen[-1]["maxInner"], 15)
            det.fitTiles(1, 2, config=dict(maxOuter=1), outDir=self.out, tile=200, pad=20,
                         visitImage=self.syn.visitImage, region=(0, 150, 0, 150), overwrite=True,
                         maxOuter=2)
            self.assertEqual(seen[-1]["maxOuter"], 2)           # an explicit override wins

    def testPlot(self):
        self._fit()
        m = det.stitchTiles(self.out, verbose=False)
        png = os.path.join(self.out, "view.png")
        st = det.plotDetector(m, out=png)
        self.assertTrue(os.path.getsize(png) > 1000)
        self.assertLess(abs(st["residualSigma"] - 1), 0.05)
        st = det.plotDetector(self.out, view=(100, 300, 50, 250), binning=1, sources=True,
                              grayscale=True)
        self.assertEqual(st["binning"], 1)

    def testTile(self):
        files = self._fit(region=(0, 150, 0, 150))                  # the corner tile
        t = det.loadTile(files[0])
        ix0, iy0, ix1, iy1 = t["interiors"][0]
        self.assertEqual((ix0, iy0), (0, 0))
        self.assertEqual(t["data"].shape[1], ix1 - ix0 + 1 + 20)    # interior + pad on one side
        self.assertEqual(int(np.sum(t["catalog"]["owned"])),
                         int(np.sum((self.syn.xy[:, 0] < ix1 + .5) & (self.syn.xy[:, 1] < iy1 + .5))))
        png = os.path.join(self.out, "tile.png")
        st = det.plotTile(files[0], out=png)
        self.assertTrue(os.path.getsize(png) > 1000)
        self.assertLess(abs(st["residualSigma"] - 1), 0.1)

    def testResidualsBlankOnBadPixels(self):
        """RESIDUAL and RESIDUAL_SIGMA are NaN on masked pixels (DATA / MODEL unchanged there)."""
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model = np.array([[9.0, 5.0], [30.0, 38.0]])
        var = np.array([[1.0, 4.0], [0.0, 4.0]])
        bad = np.array([[0, 1], [0, 0]], np.uint8)
        r, z = det._residuals(data, model, var, bad)
        np.testing.assert_array_equal(np.isnan(r), [[False, True], [False, False]])
        np.testing.assert_array_equal(np.isnan(z), [[False, True], [True, False]])
        self.assertEqual((r[0, 0], r[1, 1], z[1, 1]), (1.0, 2.0, 1.0))

    def testResidualImages(self):
        from astropy.io import fits
        files = self._fit(region=(0, 150, 0, 150))                  # the corner tile
        with fits.open(files[0]) as hl:
            names = [h.name for h in hl]
            d, m, v = (hl[k].data.astype(float) for k in ("DATA", "MODEL", "VARIANCE"))
            r, z = hl["RESIDUAL"].data, hl["RESIDUAL_SIGMA"].data
        self.assertEqual(names[1:7], ["DATA", "MODEL", "RESIDUAL", "RESIDUAL_SIGMA", "VARIANCE",
                                      "BAD"])
        np.testing.assert_allclose(r, d - m, rtol=1e-5, atol=1e-3)
        ok = np.isfinite(z)
        np.testing.assert_allclose(z[ok], ((d - m) / np.sqrt(v))[ok], rtol=1e-4, atol=1e-4)
        det.stitchTiles(self.out, verbose=False)
        with fits.open(os.path.join(self.out, "detector_mosaic.fits")) as hl:
            self.assertEqual([h.name for h in hl][1:], ["DATA", "MODEL", "RESIDUAL",
                                                        "RESIDUAL_SIGMA", "VARIANCE", "BAD",
                                                        "TILE"])
            self.assertLess(abs(np.nanstd(hl["RESIDUAL_SIGMA"].data) - 1), 0.1)
        self.assertIn("residSigma", det.loadMosaic(self.out))
        self.assertIn("residSigma", det.loadTile(files[0]))

    def testOldTilesGetResiduals(self):
        from astropy.io import fits
        files = self._fit(region=(0, 150, 0, 150))
        with fits.open(files[0]) as hl:                             # a tile from an older version
            old = fits.HDUList([h.copy() for h in hl if h.name not in ("RESIDUAL", "RESIDUAL_SIGMA")])
        old.writeto(files[0], overwrite=True)
        self.assertEqual(det.addTileResiduals(self.out), files)
        self.assertEqual(det.addTileResiduals(self.out), [])         # already there: untouched
        with fits.open(files[0]) as hl:
            self.assertEqual([h.name for h in hl][1:8], ["DATA", "MODEL", "RESIDUAL",
                                                         "RESIDUAL_SIGMA", "VARIANCE", "BAD",
                                                         "CATALOG"])
        self.assertEqual(len(det.stitchTiles(self.out, verbose=False)["catalog"]),
                         int(np.sum((self.syn.xy[:, 0] < 199.5) & (self.syn.xy[:, 1] < 166.5))))

    def testCli(self):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), "..", "bin.src", "crowdDetector.py")
        spec = importlib.util.spec_from_file_location("crowdDetector", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        self._fit()
        self.assertEqual(cli.main(["stitch", "--out", self.out]), 0)
        self.assertEqual(cli.main(["plot", "--out", self.out, "--view", "0", "200", "0", "200"]), 0)
        self.assertTrue(os.path.exists(os.path.join(self.out, "detector_view.png")))


class CpuLimitTestCase(unittest.TestCase):
    """availableCpus honours container CPU quotas, which do not hide the node's cores."""

    def _root(self, files):
        tmp = tempfile.mkdtemp()
        for rel, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            with open(os.path.join(tmp, rel), "w") as fh:
                fh.write(text)
        return tmp

    def testQuota(self):
        with mock.patch.object(os, "sched_getaffinity", lambda pid: set(range(128))):
            self.assertEqual(det.availableCpus(self._root({"cpu.max": "400000 100000\n"})), 4)
            self.assertEqual(det.availableCpus(self._root({"cpu.max": "max 100000\n"})), 128)
            self.assertEqual(det.availableCpus(self._root({"cpu/cpu.cfs_quota_us": "250000",
                                                           "cpu/cpu.cfs_period_us": "100000"})), 2)
            self.assertEqual(det.availableCpus(self._root({"cpu/cpu.cfs_quota_us": "-1",
                                                           "cpu/cpu.cfs_period_us": "100000"})), 128)
            self.assertEqual(det.availableCpus(self._root({})), 128)
            self.assertEqual(det.availableCpus(self._root({"cpu.max": "50000 100000"})), 1)
        with mock.patch.object(os, "sched_getaffinity", lambda pid: {0, 1}):
            self.assertEqual(det.availableCpus(self._root({"cpu.max": "800000 100000"})), 2)

    def testLimitBlasThreads(self):
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
        except ImportError:
            self.skipTest("threadpoolctl not installed")
        saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                 "MKL_NUM_THREADS")}
        try:
            with threadpool_limits():                              # restored afterwards
                self.assertEqual(det.limitBlasThreads(2, verbose=False), 2)
                self.assertTrue(all(p["num_threads"] <= 2 for p in threadpool_info()))
                self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "2")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
