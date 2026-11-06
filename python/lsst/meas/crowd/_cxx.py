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

"""Load the C++ extension (``_crowdLib``) and check that it was built from this source tree.

A stale build is easy to end up with: ``python setup.py build_ext --inplace`` writes
``_crowdLib.cpython-<ver>-<arch>.so`` and ``scons`` its own file next to it, and Python always
loads the tagged one, so rebuilding with the other tool changes nothing.  The extension reports
the version of its interface; a mismatch raises an error naming the files to remove.
"""

__all__ = ["API_VERSION", "StaleExtensionError", "load"]

import glob
import importlib
import os

API_VERSION = 3          # bump with every change to the extension's functions or arguments


class StaleExtensionError(ImportError):
    """The compiled extension does not match the Python code (rebuild it)."""


def load():
    """Return the C++ extension module, or None if it is not built.

    Returns
    -------
    module : `module` or `None`
        ``lsst.meas.crowd._crowdLib``.

    Raises
    ------
    StaleExtensionError
        If the module loaded was built from an older (or newer) source tree.
    """
    try:
        lib = importlib.import_module("lsst.meas.crowd._crowdLib")
    except StaleExtensionError:
        raise
    except ImportError:
        return None
    have = getattr(lib, "API_VERSION", 0)
    if have != API_VERSION:
        here = os.path.dirname(os.path.abspath(__file__))
        files = sorted(glob.glob(os.path.join(here, "_crowdLib*.so")))
        raise StaleExtensionError(
            f"the compiled C++ extension {getattr(lib, '__file__', '_crowdLib')} has interface "
            f"version {have}, this code needs {API_VERSION}: it was built from older sources.  "
            f"Python loads the first of these files it finds: {', '.join(files) or '(none)'}.  "
            f"Remove them all and rebuild once (`rm -f {os.path.join(here, '_crowdLib*.so')}`, then "
            f"`scons` or `python setup.py build_ext --inplace`), and restart the Python kernel.")
    return lib
