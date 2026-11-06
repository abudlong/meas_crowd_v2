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
// Sparse joint-fit solver.  The linear solve follows lsst-dm/pipe_crowd's
// CrowdedFieldMatrix::solve (Eigen::LeastSquaresConjugateGradient on the sparse design matrix);
// an active-set loop around it adds the x >= 0 bound meas_crowd's joint fit needs.
#include "lsst/meas/crowd/sparseSolver.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>

namespace lsst {
namespace meas {
namespace crowd {

namespace {

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, std::int32_t>;
using SpMap = Eigen::Map<SpMat const>;
using Vec = Eigen::VectorXd;
using Solver = Eigen::LeastSquaresConjugateGradient<SpMat>;

// The columns `cols` of A as their own compressed matrix (storage held in `outer`/`inner`/`vals`).
SpMap selectColumns(SpMap const &A, std::vector<int> const &cols, std::vector<std::int32_t> &outer,
                    std::vector<std::int32_t> &inner, std::vector<double> &vals) {
    std::int32_t const *ip = A.outerIndexPtr();
    std::size_t nnz = 0;
    for (int j : cols) nnz += ip[j + 1] - ip[j];
    outer.assign(cols.size() + 1, 0);
    inner.resize(nnz);
    vals.resize(nnz);
    std::size_t k = 0;
    for (std::size_t c = 0; c < cols.size(); ++c) {
        int const j = cols[c];
        std::size_t const n = ip[j + 1] - ip[j];
        std::copy(A.innerIndexPtr() + ip[j], A.innerIndexPtr() + ip[j] + n, inner.begin() + k);
        std::copy(A.valuePtr() + ip[j], A.valuePtr() + ip[j] + n, vals.begin() + k);
        k += n;
        outer[c + 1] = static_cast<std::int32_t>(k);
    }
    return SpMap(A.rows(), static_cast<Eigen::Index>(cols.size()), static_cast<Eigen::Index>(nnz),
                 outer.data(), inner.data(), vals.data());
}

}  // namespace

SparseSolveResult solveSparseLeastSquares(std::size_t nRows, std::size_t nCols,
                                          std::int32_t const *indptr, std::int32_t const *indices,
                                          double const *values, double const *b, double const *x0,
                                          bool nonNegative, double tol, int maxIter,
                                          int maxActiveSet, double kktTol,
                                          std::uint8_t const *unbounded) {
    auto bounded = [&](std::size_t j) { return nonNegative && !(unbounded && unbounded[j]); };
    SparseSolveResult out;
    out.x.assign(nCols, 0.0);
    if (nCols == 0) {
        out.converged = true;
        return out;
    }
    SpMap const A(static_cast<Eigen::Index>(nRows), static_cast<Eigen::Index>(nCols),
                  static_cast<Eigen::Index>(indptr[nCols]), indptr, indices, values);
    Eigen::Map<Vec const> const bv(b, static_cast<Eigen::Index>(nRows));

    // Columns without any pixel cannot be constrained: keep them at 0, out of the solve.
    std::vector<char> usable(nCols, 0), isFree(nCols, 0);
    for (std::size_t j = 0; j < nCols; ++j) {
        double s = 0.0;
        for (std::int32_t k = indptr[j]; k < indptr[j + 1]; ++k) s += values[k] * values[k];
        usable[j] = s > 0.0;
    }
    Vec x = Vec::Zero(static_cast<Eigen::Index>(nCols));
    for (std::size_t j = 0; j < nCols; ++j) {
        if (!usable[j]) continue;
        double const v = x0 ? x0[j] : 0.0;
        x[j] = (bounded(j) && v < 0.0) ? 0.0 : v;
        isFree[j] = 1;
    }

    Vec const atb = A.transpose() * bv;
    double const gradScale = atb.cwiseAbs().maxCoeff();
    if (gradScale == 0.0) {  // b is orthogonal to every column: x = 0 is the answer
        out.converged = true;
        return out;
    }

    std::vector<std::int32_t> outer, inner;
    std::vector<double> vals;
    std::vector<int> cols;
    int const nSolves = nonNegative ? std::max(1, maxActiveSet) : 1;
    bool allSolvesConverged = true;

    for (int it = 0; it < nSolves; ++it) {
        cols.clear();
        for (std::size_t j = 0; j < nCols; ++j)
            if (isFree[j]) cols.push_back(static_cast<int>(j));
        if (cols.empty()) {
            x.setZero();
        } else {
            bool const all = cols.size() == nCols;
            SpMap const AF = all ? A : selectColumns(A, cols, outer, inner, vals);
            Vec guess(static_cast<Eigen::Index>(cols.size()));
            for (std::size_t c = 0; c < cols.size(); ++c) guess[c] = x[cols[c]];

            Solver lscg;
            lscg.setTolerance(tol);
            lscg.setMaxIterations(maxIter);
            lscg.compute(AF);
            Vec const z = lscg.solveWithGuess(bv, guess);
            out.cgIterations += static_cast<int>(lscg.iterations());
            out.cgError = lscg.error();
            allSolvesConverged &= lscg.info() == Eigen::Success;
            out.activeSetIterations += 1;

            bool anyNegative = false;
            for (std::size_t c = 0; c < cols.size(); ++c) {
                if (bounded(static_cast<std::size_t>(cols[c])) && z[c] < 0.0) {
                    x[cols[c]] = 0.0;
                    isFree[cols[c]] = 0;
                    anyNegative = true;
                } else {
                    x[cols[c]] = z[c];
                }
            }
            if (!nonNegative) {
                out.converged = allSolvesConverged;
                break;
            }
            if (anyNegative) continue;  // re-solve with those columns held at zero
        }
        // All free values are >= 0: free the held columns the fit would push above zero.
        Vec const grad = A.transpose() * (bv - A * x);
        bool released = false;
        for (std::size_t j = 0; j < nCols; ++j) {
            if (usable[j] && !isFree[j] && grad[j] > kktTol * gradScale) {
                isFree[j] = 1;
                released = true;
            }
        }
        if (!released) {
            out.converged = allSolvesConverged;
            break;
        }
    }

    for (std::size_t j = 0; j < nCols; ++j) {
        out.x[j] = x[j];
        if (nonNegative && usable[j] && !isFree[j]) out.nAtBound += 1;
    }
    return out;
}

}  // namespace crowd
}  // namespace meas
}  // namespace lsst
