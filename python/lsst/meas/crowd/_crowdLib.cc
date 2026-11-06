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
#include <pybind11/stl.h>

#include "lsst/meas/crowd/centroids.h"
#include "lsst/meas/crowd/jointFit.h"
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

void pyCropStamps(py::array_t<double, py::array::c_style> &stamps, IArray const &x0,
                  IArray const &y0, Array const &xs, Array const &ys, IArray const &halves) {
    if (stamps.ndim() != 3) throw std::invalid_argument("stamps must be 3-d");
    std::size_t const n = stamps.shape(0);
    if (static_cast<std::size_t>(x0.size()) != n || static_cast<std::size_t>(y0.size()) != n ||
        static_cast<std::size_t>(xs.size()) != n || static_cast<std::size_t>(ys.size()) != n ||
        static_cast<std::size_t>(halves.size()) != n)
        throw std::invalid_argument("x0 / y0 / xs / ys / halves must have one value per stamp");
    py::gil_scoped_release release;
    cropStamps(stamps.mutable_data(), n, stamps.shape(1), stamps.shape(2), x0.data(), y0.data(),
               xs.data(), ys.data(), halves.data());
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

using I64Array = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

// Stamps from Python: a list with, per source, None or (2-d float64 array, x0, y0) with (x0, y0)
// the stamp origin in image-array coordinates.  Strided arrays (views) are used as they are; the
// converted arrays are kept in `keep` for the duration of the call.
std::vector<StampView> stampViews(py::list const &stamps, std::vector<py::array_t<double>> &keep) {
    std::vector<StampView> views(stamps.size());
    keep.reserve(stamps.size());
    for (std::size_t j = 0; j < stamps.size(); ++j) {
        py::handle h = stamps[j];
        if (h.is_none()) continue;
        py::tuple t = h.cast<py::tuple>();
        keep.push_back(py::array_t<double>::ensure(t[0]));
        py::array_t<double> const &a = keep.back();
        if (!a || a.ndim() != 2) throw std::invalid_argument("each stamp must be a 2-d float array");
        StampView v;
        v.data = a.data();
        v.ny = a.shape(0);
        v.nx = a.shape(1);
        v.rowStride = a.strides(0) / static_cast<long>(sizeof(double));
        v.colStride = a.strides(1) / static_cast<long>(sizeof(double));
        v.x0 = t[1].cast<long>();
        v.y0 = t[2].cast<long>();
        views[j] = v;
    }
    return views;
}

py::dict pyBuildDesignMatrix(py::list const &stamps, I64Array const &pixelIndex, Array const &sqrtW,
                             double supportFrac) {
    if (pixelIndex.ndim() != 2) throw std::invalid_argument("pixelIndex must be 2-d");
    std::vector<py::array_t<double>> keep;
    std::vector<StampView> views = stampViews(stamps, keep);
    DesignMatrix m;
    {
        py::gil_scoped_release release;
        m = buildDesignMatrix(views, pixelIndex.data(), pixelIndex.shape(0), pixelIndex.shape(1),
                              sqrtW.data(), static_cast<long>(sqrtW.size()), supportFrac);
    }
    auto toArray = [](auto const &v) {
        using T = typename std::decay_t<decltype(v)>::value_type;
        py::array_t<T> a(v.size());
        std::copy(v.begin(), v.end(), a.mutable_data());
        return a;
    };
    py::dict out;
    out["indptr"] = toArray(m.indptr);
    out["indices"] = toArray(m.indices);
    out["data"] = toArray(m.data);
    out["labels"] = toArray(m.labels);
    out["nGroups"] = m.nGroups;
    out["sig"] = toArray(m.sig);
    out["sigOffset"] = toArray(m.sigOffset);
    return out;
}

py::tuple pyAccumulateModel(py::list const &stamps, Array const &amps, double ampZeroTol,
                            py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> const &labels,
                            py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> const &sig,
                            I64Array const &sigOffset, long ny, long nx) {
    std::vector<py::array_t<double>> keep;
    std::vector<StampView> views = stampViews(stamps, keep);
    if (static_cast<std::size_t>(amps.size()) != views.size() ||
        static_cast<std::size_t>(labels.size()) != views.size() ||
        static_cast<std::size_t>(sigOffset.size()) != views.size() + 1)
        throw std::invalid_argument("amps / labels / sigOffset do not match the stamps");
    Array model({ny, nx});
    I64Array groupMap({ny, nx});
    std::fill(model.mutable_data(), model.mutable_data() + ny * nx, 0.0);
    std::fill(groupMap.mutable_data(), groupMap.mutable_data() + ny * nx, std::int64_t(-1));
    {
        py::gil_scoped_release release;
        accumulateModel(views, amps.data(), ampZeroTol, labels.data(), sig.data(), sigOffset.data(),
                        ny, nx, model.mutable_data(), groupMap.mutable_data());
    }
    return py::make_tuple(model, groupMap);
}

py::tuple pyRefineCentroids(Array const &kernel, int order, double eps, Array const &xs,
                            Array const &ys, Array const &amps, py::list const &stamps,
                            Array const &resid, Array const &var,
                            py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> const &good,
                            long x0Img, long y0Img, double R2, long rWin, double maxShift) {
    check2d(kernel, "kernel");
    check2d(resid, "resid");
    long const ny = resid.shape(0), nx = resid.shape(1);
    if (var.size() != ny * nx || good.size() != ny * nx)
        throw std::invalid_argument("var / good must match resid");
    std::vector<py::array_t<double>> keep;
    std::vector<StampView> views = stampViews(stamps, keep);
    std::size_t const n = views.size();
    if (static_cast<std::size_t>(xs.size()) != n || static_cast<std::size_t>(ys.size()) != n ||
        static_cast<std::size_t>(amps.size()) != n)
        throw std::invalid_argument("one position, flux and stamp per source expected");
    std::vector<CentroidSource> src(n);
    for (std::size_t j = 0; j < n; ++j) {
        src[j].x = xs.data()[j];
        src[j].y = ys.data()[j];
        src[j].amp = amps.data()[j];
        src[j].stamp = views[j];
    }
    Array newX(n), newY(n);
    py::array_t<std::uint8_t> moved(n);
    {
        py::gil_scoped_release release;
        refineCentroids(kernel.data(), kernel.shape(0), kernel.shape(1), order, eps, src,
                        resid.data(), var.data(), good.data(), ny, nx, x0Img, y0Img, R2, rWin,
                        maxShift, newX.mutable_data(), newY.mutable_data(), moved.mutable_data());
    }
    return py::make_tuple(newX, newY, moved);
}

py::tuple pyRenderDerivativesWindow(Array const &kernel, double x, double y, int order, double eps,
                                    long r0, long r1, long c0, long c1) {
    check2d(kernel, "kernel");
    long const ny = kernel.shape(0), nx = kernel.shape(1);
    if (r0 < 0 || c0 < 0 || r1 > ny || c1 > nx || r0 >= r1 || c0 >= c1)
        throw std::invalid_argument("window outside the kernel");
    Array dx({r1 - r0, c1 - c0}), dy({r1 - r0, c1 - c0});
    {
        py::gil_scoped_release release;
        renderDerivativesWindow(kernel.data(), ny, nx, x, y, order, eps, r0, r1, c0, c1,
                                dx.mutable_data(), dy.mutable_data());
    }
    return py::make_tuple(dx, dy);
}

py::dict pySolveSparseLeastSquares(std::size_t nRows, std::size_t nCols, I32Array const &indptr,
                                   I32Array const &indices, Array const &data, Array const &b,
                                   py::object const &x0, bool nonNegative, double tol, int maxIter,
                                   int maxActiveSet, double kktTol, py::object const &unbounded) {
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
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> unb;
    std::uint8_t const *unbp = nullptr;
    if (!unbounded.is_none()) {
        unb = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::ensure(unbounded);
        if (!unb || static_cast<std::size_t>(unb.size()) != nCols)
            throw std::invalid_argument("unbounded must have nCols entries");
        unbp = unb.data();
    }
    SparseSolveResult res;
    {
        py::gil_scoped_release release;
        res = solveSparseLeastSquares(nRows, nCols, indptr.data(), indices.data(), data.data(),
                                      b.data(), x0p, nonNegative, tol, maxIter, maxActiveSet,
                                      kktTol, unbp);
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
    mod.def("renderDerivativesWindow", &pyRenderDerivativesWindow, "kernel"_a, "x"_a, "y"_a,
            "order"_a, "eps"_a, "r0"_a, "r1"_a, "c0"_a, "c1"_a);
    mod.def("cropStamps", &pyCropStamps, "stamps"_a, "x0"_a, "y0"_a, "xs"_a, "ys"_a, "halves"_a,
            "Crop stamps in place beyond a radius per stamp (cosine taper over the outer quarter).");
    mod.def("refineCentroids", &pyRefineCentroids, "kernel"_a, "order"_a, "eps"_a, "xs"_a, "ys"_a,
            "amps"_a, "stamps"_a, "resid"_a, "var"_a, "good"_a, "x0Img"_a, "y0Img"_a, "R2"_a,
            "rWin"_a, "maxShift"_a);
    mod.def("buildDesignMatrix", &pyBuildDesignMatrix, "stamps"_a, "pixelIndex"_a, "sqrtW"_a,
            "supportFrac"_a);
    mod.def("accumulateModel", &pyAccumulateModel, "stamps"_a, "amps"_a, "ampZeroTol"_a,
            "labels"_a, "sig"_a, "sigOffset"_a, "ny"_a, "nx"_a);
    mod.def("solveSparseLeastSquares", &pySolveSparseLeastSquares, "nRows"_a, "nCols"_a,
            "indptr"_a, "indices"_a, "data"_a, "b"_a, "x0"_a = py::none(), "nonNegative"_a = true,
            "tol"_a = 1e-8, "maxIter"_a = 5000, "maxActiveSet"_a = 50, "kktTol"_a = 1e-6,
            "unbounded"_a = py::none(),
            "Sparse (non-negative) least squares: Eigen LeastSquaresConjugateGradient, as in "
            "pipe_crowd, inside an active-set loop for x >= 0.  A is given in CSC form.");
}

}  // namespace crowd
}  // namespace meas
}  // namespace lsst
