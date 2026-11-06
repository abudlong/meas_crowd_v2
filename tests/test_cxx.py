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

"""Loading the C++ extension: a stale build is reported, a missing one is not an error."""

import importlib
import types
import unittest
from unittest import mock

import lsstStubs

lsstStubs.install()

from lsst.meas.crowd import _cxx  # noqa: E402


class LoadTestCase(unittest.TestCase):
    def testCurrentBuild(self):
        lib = _cxx.load()
        if lib is None:
            self.skipTest("C++ extension not built")
        self.assertEqual(lib.API_VERSION, _cxx.API_VERSION)
        self.assertTrue(hasattr(lib, "solveSparseLeastSquares"))

    def testStaleBuildIsReported(self):
        stale = types.ModuleType("lsst.meas.crowd._crowdLib")       # an old build: no version
        stale.__file__ = "/somewhere/_crowdLib.cpython-313-x86_64-linux-gnu.so"
        with mock.patch.object(importlib, "import_module", lambda name: stale):
            with self.assertRaises(_cxx.StaleExtensionError) as cm:
                _cxx.load()
        msg = str(cm.exception)
        self.assertIn("rebuild", msg)
        self.assertIn("_crowdLib*.so", msg)
        self.assertIn(stale.__file__, msg)
        self.assertTrue(issubclass(_cxx.StaleExtensionError, ImportError))

    def testMissingBuild(self):
        def missing(name):
            raise ModuleNotFoundError(name)
        with mock.patch.object(importlib, "import_module", missing):
            self.assertIsNone(_cxx.load())


if __name__ == "__main__":
    unittest.main()
