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
#ifndef LSST_MEAS_CROWD_SPARSESOLVER_H
#define LSST_MEAS_CROWD_SPARSESOLVER_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lsst {
namespace meas {
namespace crowd {

/// What `solveSparseLeastSquares` returns.
struct SparseSolveResult {
    std::vector<double> x;       ///< solution, one value per column
    int cgIterations = 0;        ///< conjugate-gradient iterations, summed over all solves
    int activeSetIterations = 0; ///< number of LSCG solves (1 without the x >= 0 bound)
    int nAtBound = 0;            ///< columns held at zero by the x >= 0 bound
    bool converged = false;      ///< every LSCG solve converged and the KKT conditions hold
    double cgError = 0.0;        ///< relative normal-equation residual of the last solve
};

/**
 * Solve min ||A x - b||^2 for a sparse A, optionally with x >= 0.
 *
 * The linear solve is the one used by lsst-dm/pipe_crowd (CrowdedFieldMatrix::solve):
 * Eigen's LeastSquaresConjugateGradient, i.e. conjugate gradients on the normal equations with
 * a Jacobi (column-norm) preconditioner, working directly on the sparse (pixels x sources)
 * matrix, so A^T A is never formed.
 *
 * pipe_crowd's solve is unconstrained.  For nonNegative=true it is wrapped in an active-set
 * loop: solve on the free columns; columns that come out negative are fixed at zero and the
 * free set re-solved; once all free values are >= 0, a fixed column whose gradient A_j^T (b - Ax)
 * is positive (the fit would like it > 0) is freed again.  Stops when nothing changes (the KKT
 * conditions of the bounded problem) or after maxActiveSet solves.
 *
 * A is given in compressed-sparse-column form (scipy ``csc_matrix``: indptr, indices, data).
 *
 * @param nRows, nCols   shape of A
 * @param indptr         column pointers, length nCols + 1
 * @param indices        row index of every entry, length indptr[nCols]
 * @param values         value of every entry
 * @param b              right-hand side, length nRows
 * @param x0             starting guess (length nCols) or nullptr for zeros
 * @param nonNegative    enforce x >= 0
 * @param tol            LSCG tolerance: ||A^T r|| / ||A^T b|| (pipe_crowd uses 1e-6)
 * @param maxIter        LSCG iteration cap per solve (pipe_crowd uses 500)
 * @param maxActiveSet   cap on the number of LSCG solves of the active-set loop
 * @param kktTol         a fixed column is freed when A_j^T r > kktTol * ||A^T b||_inf
 * @param unbounded      per column, non-zero = exempt from x >= 0 (e.g. sky terms); nullptr = none
 */
SparseSolveResult solveSparseLeastSquares(std::size_t nRows, std::size_t nCols,
                                          std::int32_t const *indptr, std::int32_t const *indices,
                                          double const *values, double const *b, double const *x0,
                                          bool nonNegative, double tol, int maxIter,
                                          int maxActiveSet, double kktTol,
                                          std::uint8_t const *unbounded = nullptr);

}  // namespace crowd
}  // namespace meas
}  // namespace lsst

#endif  // LSST_MEAS_CROWD_SPARSESOLVER_H
