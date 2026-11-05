/*
 * This file is part of meas_crowd.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "lsst/meas/crowd/sparseSolver.h"
#include "lsst/meas/crowd/stamps.h"

namespace py = pybind11;
using namespace pybind11::literals;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<long>;
using I32Array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

namespace lsst {
namespace meas {
namespace crowd {
namespace {

void check2d(Array const &a, char const *name) {
    if (a.ndim() != 2) throw std::invalid_argument(std::string(name) + " must be 2-d");
}

Array pyShiftImage(Array const &image, double dx, double dy, int order) {
    check2d(image, "image");
    std::size_t const ny = image.shape(0), nx = image.shape(1);
    Array out({ny, nx});
    {
        py::gil_scoped_release release;
        shiftImage(image.data(), ny, nx, dx, dy, order, out.mutable_data());
    }
    return out;
}

Array pyTaperStamp(Array const &stamp, double cx, double cy, double t0, double t1) {
    check2d(stamp, "stamp");
    std::size_t const ny = stamp.shape(0), nx = stamp.shape(1);
    Array out({ny, nx});
    std::copy(stamp.data(), stamp.data() + ny * nx, out.mutable_data());
    taperStamp(out.mutable_data(), ny, nx, cx, cy, t0, t1);
    return out;
}

py::tuple pyRenderStamps(Array const &kernel, Array const &xs, Array const &ys, int order,
                         bool taper, double t0, double t1) {
    check2d(kernel, "kernel");
    if (xs.size() != ys.size()) throw std::invalid_argument("xs and ys differ in length");
    std::size_t const ny = kernel.shape(0), nx = kernel.shape(1), n = xs.size();
    Array out({n, ny, nx});
    IArray x0(n), y0(n);
    {
        py::gil_scoped_release release;
        renderStamps(kernel.data(), ny, nx, xs.data(), ys.data(), n, order, taper, t0, t1,
                     out.mutable_data(), x0.mutable_data(), y0.mutable_data());
    }
    return py::make_tuple(out, x0, y0);
}

py::tuple pyRenderDerivatives(Array const &kernel, Array const &xs, Array const &ys, int order,
                              double eps) {
    check2d(kernel, "kernel");
    if (xs.size() != ys.size()) throw std::invalid_argument("xs and ys differ in length");
    std::size_t const ny = kernel.shape(0), nx = kernel.shape(1), n = xs.size();
    Array dx({n, ny, nx}), dy({n, ny, nx});
    IArray x0(n), y0(n);
    {
        py::gil_scoped_release release;
        renderDerivatives(kernel.data(), ny, nx, xs.data(), ys.data(), n, order, eps,
                          dx.mutable_data(), dy.mutable_data(), x0.mutable_data(),
                          y0.mutable_data());
    }
    return py::make_tuple(dx, dy, x0, y0);
}

py::dict pySolveSparseLeastSquares(std::size_t nRows, std::size_t nCols, I32Array const &indptr,
                                   I32Array const &indices, Array const &data, Array const &b,
                                   py::object const &x0, bool nonNegative, double tol, int maxIter,
                                   int maxActiveSet, double kktTol) {
    if (static_cast<std::size_t>(indptr.size()) != nCols + 1)
        throw std::invalid_argument("indptr must have nCols + 1 entries");
    std::size_t const nnz = static_cast<std::size_t>(indptr.data()[nCols]);
    if (static_cast<std::size_t>(indices.size()) < nnz || static_cast<std::size_t>(data.size()) < nnz)
        throw std::invalid_argument("indices / data shorter than indptr[-1]");
    if (static_cast<std::size_t>(b.size()) != nRows)
        throw std::invalid_argument("b must have nRows entries");
    Array guess;
    double const *x0p = nullptr;
    if (!x0.is_none()) {
        guess = Array::ensure(x0);
        if (!guess || static_cast<std::size_t>(guess.size()) != nCols)
            throw std::invalid_argument("x0 must have nCols entries");
        x0p = guess.data();
    }
    SparseSolveResult res;
    {
        py::gil_scoped_release release;
        res = solveSparseLeastSquares(nRows, nCols, indptr.data(), indices.data(), data.data(),
                                      b.data(), x0p, nonNegative, tol, maxIter, maxActiveSet,
                                      kktTol);
    }
    Array x(nCols);
    std::copy(res.x.begin(), res.x.end(), x.mutable_data());
    py::dict out;
    out["x"] = x;
    out["cgIterations"] = res.cgIterations;
    out["activeSetIterations"] = res.activeSetIterations;
    out["nAtBound"] = res.nAtBound;
    out["converged"] = res.converged;
    out["cgError"] = res.cgError;
    return out;
}

}  // namespace

PYBIND11_MODULE(_crowdLib, mod) {
    mod.doc() = "Fast PSF-stamp kernels and the sparse joint-fit solver for lsst.meas.crowd";
    mod.def("shiftImage", &pyShiftImage, "image"_a, "dx"_a, "dy"_a, "order"_a = 5);
    mod.def("taperStamp", &pyTaperStamp, "stamp"_a, "cx"_a, "cy"_a, "t0"_a, "t1"_a);
    mod.def("renderStamps", &pyRenderStamps, "kernel"_a, "xs"_a, "ys"_a, "order"_a = 5,
            "taper"_a = false, "t0"_a = 0.0, "t1"_a = 1.0);
    mod.def("renderDerivatives", &pyRenderDerivatives, "kernel"_a, "xs"_a, "ys"_a, "order"_a = 5,
            "eps"_a = 0.25);
    mod.def("solveSparseLeastSquares", &pySolveSparseLeastSquares, "nRows"_a, "nCols"_a,
            "indptr"_a, "indices"_a, "data"_a, "b"_a, "x0"_a = py::none(), "nonNegative"_a = true,
            "tol"_a = 1e-8, "maxIter"_a = 5000, "maxActiveSet"_a = 50, "kktTol"_a = 1e-6,
            "Sparse (non-negative) least squares: Eigen LeastSquaresConjugateGradient, as in "
            "pipe_crowd, inside an active-set loop for x >= 0.  A is given in CSC form.");
}

}  // namespace crowd
}  // namespace meas
}  // namespace lsst
