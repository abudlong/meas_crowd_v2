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

"""The PIFF PSF refit (refitPsf="piff"): the neighbour-subtracted image and the wiring to the
stack's tasks.  PIFF itself is not run here (it needs the Rubin stack); a recording stand-in
checks what it would be given."""

import copy
import sys
import types
import unittest
from unittest import mock

import numpy as np

import lsstStubs
from lsstStubs import Box2I, Point2I

lsstStubs.install()

from lsst.meas.crowd import fitLib as L  # noqa: E402
from lsst.meas.crowd import pipeline as pl  # noqa: E402
from test_sparseSolver import _gaussStamp  # noqa: E402


def _scene(n=120, sigma=1.8, half=12, seed=2):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(15, n - 15, (40, 2))
    flux = 10 ** rng.uniform(3, 4.5, len(xy))
    stamp = _gaussStamp(sigma, half)
    box = Box2I(Point2I(0, 0), Point2I(n - 1, n - 1))
    comps, model = [], np.zeros((n, n))
    for (x, y), f in zip(xy, flux):
        a, bb = stamp(None, x, y, box)
        ys, xs = slice(bb.getMinY(), bb.getMaxY() + 1), slice(bb.getMinX(), bb.getMaxX() + 1)
        comps.append(dict(stamp=a, ys=ys, xs=xs))
        model[ys, xs] += f * a
    data = model + rng.normal(size=model.shape) * 5.0
    fit = dict(modelFull=model, comps=comps, amps=flux)
    return data, fit, xy, flux, comps


class PsfStarImageTestCase(unittest.TestCase):
    def testOnlyThePsfStarsRemain(self):
        data, fit, xy, flux, comps = _scene()
        exp = lsstStubs.FakeExposure(data.copy(), np.full(data.shape, 25.0))
        idx = [3, 17, 30]
        img = pl.psfStarImage(exp, fit, idx)
        stars = np.zeros_like(data)
        for j in idx:
            stars[comps[j]["ys"], comps[j]["xs"]] += flux[j] * comps[j]["stamp"]
        np.testing.assert_allclose(img, (data - fit["modelFull"]) + stars, atol=1e-9)
        np.testing.assert_array_equal(exp.image.array, data)            # input untouched


class _Cat(list):
    def __init__(self, schema=None):
        super().__init__()

    def addNew(self):
        rec = types.SimpleNamespace(pos=None, footprint=None)
        rec.set = lambda key, value, r=rec: setattr(r, "pos", (value[0], value[1]))
        rec.setFootprint = lambda fp, r=rec: setattr(r, "footprint", fp)
        self.append(rec)
        return rec

    def copy(self, deep=False):
        out = _Cat()
        out.extend(self)
        return out


class _Spans:
    """Stand-in for `lsst.afw.geom.SpanSet`: a disc of the given radius."""

    def __init__(self, radius, center):
        self.radius, self.center = radius, center

    @classmethod
    def fromShape(cls, radius, stencil, offset):
        return cls(radius, offset)

    def clippedTo(self, box):
        return self

    def getArea(self):
        return int(np.pi * self.radius ** 2) + 1


class _Footprint:
    """Stand-in for `lsst.afw.detection.Footprint` (spans and peaks)."""

    def __init__(self, spans, box):
        self.spans, self.peaks = spans, []

    def addPeak(self, x, y, value):
        self.peaks.append((x, y, value))


class RefitPsfPiffWiringTestCase(unittest.TestCase):
    def test(self):
        data, fit, xy, flux, comps = _scene()
        u = np.arange(-12, 13)
        K = np.exp(-0.5 * (u[None] ** 2 + u[:, None] ** 2) / 1.8 ** 2)
        psf = lsstStubs.FakePsf(K / K.sum())

        class Exp(lsstStubs.FakeExposure):
            def clone(self):
                return copy.deepcopy(self)

            def setPsf(self, p):
                self._psf = p

        exp = Exp(data.copy(), np.full(data.shape, 25.0))
        res = dict(snaps=[dict(fit=fit)], S=dict(xy=xy))
        idx = list(range(0, 40, 3))
        seen = {}

        class FakeMakeCandidates:
            ConfigClass = staticmethod(lambda: types.SimpleNamespace(kernelSize=None))

            def __init__(self, config):
                seen["candidateStamp"] = config.kernelSize

            def run(self, cat, exposure):
                seen["candidatePositions"] = [r.pos for r in cat]
                seen["footprints"] = [r.footprint for r in cat]
                seen["candidateImage"] = exposure.image.array.copy()
                return types.SimpleNamespace(psfCandidates=list(range(len(cat))))

        class FakePiff:
            ConfigClass = staticmethod(lambda: types.SimpleNamespace(spatialOrder=2, stampSize=0,
                                                                     modelSize=25))

            def __init__(self, config):
                seen["config"] = config

            def determinePsf(self, exposure, cands, metadata):
                seen["nCands"] = len(cands)
                return psf, None

        piffMod = types.ModuleType("lsst.meas.extensions.piff.piffPsfDeterminer")
        piffMod.PiffPsfDeterminerTask = FakePiff
        dafBase = types.ModuleType("lsst.daf.base")
        dafBase.PropertyList = dict
        afwTable = types.SimpleNamespace(
            SourceTable=types.SimpleNamespace(makeMinimalSchema=lambda: types.SimpleNamespace(
                getAliasMap=lambda: types.SimpleNamespace(set=lambda *a: None))),
            Point2DKey=types.SimpleNamespace(addFields=lambda *a: "centroidKey"),
            SourceCatalog=_Cat)
        P = dict(pl._params({}), piffStampSize=25, piffSpatialOrder=1, piffMinStars=5)
        mods = {"lsst.meas.extensions": types.ModuleType("lsst.meas.extensions"),
                "lsst.meas.extensions.piff": types.ModuleType("lsst.meas.extensions.piff"),
                "lsst.meas.extensions.piff.piffPsfDeterminer": piffMod, "lsst.daf.base": dafBase}
        with mock.patch.dict(sys.modules, mods), \
                mock.patch.object(pl, "selectPsfStars", lambda *a, **k: idx), \
                mock.patch.object(L, "afwTable", afwTable, create=True), \
                mock.patch.object(L, "afwGeom", types.SimpleNamespace(
                    SpanSet=_Spans, Stencil=types.SimpleNamespace(CIRCLE="circle")), create=True), \
                mock.patch.object(L, "afwDetect", types.SimpleNamespace(Footprint=_Footprint),
                                  create=True), \
                mock.patch.object(L, "MakePsfCandidatesTask", FakeMakeCandidates, create=True), \
                mock.patch.object(L, "_filterValidPsfCandidates", lambda c, verbose=True: c,
                                  create=True), \
                mock.patch.object(L, "_fwhm", lambda p, pt: 4.24, create=True):
            out = pl.refitPsfPiff(exp, psf, res, P, (60.0, 60.0), verbose=False)
            self.assertIs(out, psf)
            self.assertEqual(seen["candidateStamp"], 27)                # model 25 + 2
            self.assertEqual((seen["config"].spatialOrder, seen["config"].stampSize), (1, 27))
            self.assertEqual(seen["config"].modelSize, 25)
            self.assertEqual(seen["nCands"], len(idx))
            np.testing.assert_allclose(seen["candidatePositions"], xy[idx])
            # the stack needs a footprint with a peak on every star
            for fp, (x, y) in zip(seen["footprints"], xy[idx]):
                self.assertIsNotNone(fp)
                self.assertEqual(fp.spans.radius, 13)
                np.testing.assert_allclose(fp.peaks[0][:2], (x, y))
                c = fp.spans.center                  # a stub or an lsst.geom Point2I
                self.assertEqual((int(c[0]), int(c[1])), (int(round(x)), int(round(y))))
            np.testing.assert_allclose(seen["candidateImage"], pl.psfStarImage(exp, fit, idx),
                                       atol=1e-3)
            xyEdge = xy.copy()                           # a star whose 27 px stamp does not fit
            xyEdge[0] = (5.0, 60.0)
            seen.clear()
            pl.refitPsfPiff(exp, psf, dict(res, S=dict(xy=xyEdge)), P, (60.0, 60.0), verbose=False)
            np.testing.assert_allclose(seen["candidatePositions"], xyEdge[idx[1:]])
            seen.clear()                                 # the stamps follow the model size
            pl.refitPsfPiff(exp, psf, res, dict(P, piffStampSize=21), (60.0, 60.0), verbose=False)
            self.assertEqual((seen["config"].stampSize, seen["config"].modelSize), (23, 21))
            with mock.patch.object(pl, "selectPsfStars", lambda *a, **k: idx[:3]):
                with self.assertRaises(RuntimeError):                  # too few stars
                    pl.refitPsfPiff(exp, psf, res, P, (60.0, 60.0), verbose=False)


class PiffFailureReportTestCase(unittest.TestCase):
    """`pipeline._piffFailureReport`: what PIFF saw under its stars when it gives up."""

    def test(self):
        import contextlib
        import io
        data, fit, xy, flux, comps = _scene()
        exp = lsstStubs.FakeExposure(data.copy(), np.full(data.shape, 25.0))
        exp.mask.array[:, :60] |= exp.mask.getPlaneBitMask("BAD")      # the left half
        cfg = types.SimpleNamespace(badMaskPlanes=["BAD", "SAT"], modelSize=25)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pl._piffFailureReport(exp, xy[:10], 25, cfg)
        out = buf.getvalue()
        self.assertIn("PIFF failure report: 10 stars", out)
        self.assertIn("BAD", out)
        self.assertIn("PIFF ignores pixels with: ['BAD', 'SAT']", out)
        self.assertIn("PIFF modelSize = 25", out)

    def testPiffWhyCollectsLogging(self):
        import contextlib
        import io
        import logging

        class Task:
            def __init__(self, config):
                self.config = config

            def determinePsf(self, exposure, cands, metadata):
                log = logging.getLogger("piff.test")
                log.warning("Failed to initialize star 1: flagged")
                log.warning("Failed to initialize star 1: flagged")         # a repeat
                log.info("star 2 flagged (level %d)", self.config.piffLoggingLevel)
                raise RuntimeError("All stars are flagged.")

        class Cfg(types.SimpleNamespace):
            def validate(self):                        # this stack allows levels up to 2
                if self.piffLoggingLevel > 2:
                    raise ValueError("piffLoggingLevel out of range")

        cfg = Cfg(piffLoggingLevel=0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pl._piffWhy(Task, cfg, None, [], dict)
        out = buf.getvalue()
        self.assertIn("2 distinct; rerun failed: RuntimeError", out)
        self.assertIn("piff.test WARNING: Failed to initialize star 1", out)
        self.assertIn("star 2 flagged (level 2)", out)
        self.assertEqual(logging.getLogger().handlers.count(None), 0)

    def testNeverRaises(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pl._piffFailureReport(None, np.zeros((1, 2)), 25, None)
        self.assertIn("report failed", buf.getvalue())


class RefitPsfOptionTestCase(unittest.TestCase):
    """PIFF is the pipeline's only PSF refit."""

    def testValues(self):
        self.assertEqual(pl._params({})["refitPsf"], "piff")
        self.assertEqual(pl._params(dict(refitPsf=True))["refitPsf"], "piff")
        self.assertFalse(pl._params(dict(refitPsf=False))["refitPsf"])
        for gone in ("stack", "pca"):
            with self.assertRaises(ValueError):
                pl._params(dict(refitPsf=gone))

    def testPresets(self):
        from lsst.meas.crowd import session
        for cfg in (session.CROWDED_FIELD_CONFIG, session.CROWDED_STELLAR_FIELD_CONFIG,
                    session.FAST_STELLAR_FIELD_CONFIG):
            self.assertEqual(pl._params(session.makeConfig(cfg))["refitPsf"], "piff")

    def testRemovedRefits(self):
        for name in ("refinePsfClean", "refinePsfSpatial"):
            self.assertFalse(hasattr(pl, name))


if __name__ == "__main__":
    unittest.main()
