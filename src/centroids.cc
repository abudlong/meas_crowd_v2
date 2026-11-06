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
// The per-source loop of the centroid refinement (pipeline._refineSubset) in C++.
#include "lsst/meas/crowd/centroids.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "lsst/meas/crowd/stamps.h"

namespace lsst {
namespace meas {
namespace crowd {

void refineCentroids(double const *kernel, long kny, long knx, int order, double eps,
                     std::vector<CentroidSource> const &sources, double const *resid,
                     double const *var, std::uint8_t const *good, long ny, long nx, long x0Img,
                     long y0Img, double R2, long rWin, double maxShift, double *newX, double *newY,
                     std::uint8_t *moved) {
    long const x1Img = x0Img + nx - 1, y1Img = y0Img + ny - 1;
    std::vector<double> dx, dy;
    Eigen::MatrixX3d J;
    Eigen::VectorXd rhs;
    for (std::size_t j = 0; j < sources.size(); ++j) {
        CentroidSource const &s = sources[j];
        newX[j] = s.x;
        newY[j] = s.y;
        moved[j] = 0;
        if (s.stamp.data == nullptr) continue;
        // window around the source, clipped to the image, then to the kernel stamp
        long const ix = static_cast<long>(std::floor(s.x)), iy = static_cast<long>(std::floor(s.y));
        long wx0 = std::max(ix - rWin, x0Img), wx1 = std::min(ix + rWin + 1, x1Img);
        long wy0 = std::max(iy - rWin, y0Img), wy1 = std::min(iy + rWin + 1, y1Img);
        if (wx0 > wx1 || wy0 > wy1) continue;
        long const sx0 = static_cast<long>(std::floor(s.x + 0.5)) - knx / 2;
        long const sy0 = static_cast<long>(std::floor(s.y + 0.5)) - kny / 2;
        wx0 = std::max(wx0, sx0);
        wy0 = std::max(wy0, sy0);
        wx1 = std::min(wx1, sx0 + knx - 1);
        wy1 = std::min(wy1, sy0 + kny - 1);
        if (wx0 > wx1 || wy0 > wy1) continue;
        long const h = wy1 - wy0 + 1, w = wx1 - wx0 + 1;
        // the fitted stamp must cover the window (as pipeline's shape check)
        long const px0 = x0Img + s.stamp.x0, py0 = y0Img + s.stamp.y0;
        if (wx0 < px0 || wy0 < py0 || wx1 > px0 + s.stamp.nx - 1 || wy1 > py0 + s.stamp.ny - 1)
            continue;
        // good pixels within sqrt(R2)
        long nGood = 0;
        for (long r = 0; r < h; ++r) {
            double const ddy = static_cast<double>(wy0 + r) - s.y;
            for (long c = 0; c < w; ++c) {
                double const ddx = static_cast<double>(wx0 + c) - s.x;
                long const g = (wy0 + r - y0Img) * nx + (wx0 + c - x0Img);
                if (good[g] && ddx * ddx + ddy * ddy < R2) ++nGood;
            }
        }
        if (nGood < 10) continue;
        dx.resize(h * w);
        dy.resize(h * w);
        renderDerivativesWindow(kernel, kny, knx, s.x, s.y, order, eps, wy0 - sy0, wy1 - sy0 + 1,
                                wx0 - sx0, wx1 - sx0 + 1, dx.data(), dy.data());
        J.resize(nGood, 3);
        rhs.resize(nGood);
        long q = 0;
        for (long r = 0; r < h; ++r) {
            double const ddy = static_cast<double>(wy0 + r) - s.y;
            for (long c = 0; c < w; ++c) {
                double const ddx = static_cast<double>(wx0 + c) - s.x;
                long const g = (wy0 + r - y0Img) * nx + (wx0 + c - x0Img);
                if (!(good[g] && ddx * ddx + ddy * ddy < R2)) continue;
                double const wt = 1.0 / std::sqrt(var[g]);
                double const pm = s.stamp.data[(wy0 + r - py0) * s.stamp.rowStride +
                                               (wx0 + c - px0) * s.stamp.colStride];
                J(q, 0) = pm * wt;
                J(q, 1) = (s.amp * dx[r * w + c]) * wt;
                J(q, 2) = (s.amp * dy[r * w + c]) * wt;
                rhs[q] = resid[g] * wt;
                ++q;
            }
        }
        // minimum-norm least squares; numpy's default cut-off: eps * max(rows, cols) * s_max
        Eigen::JacobiSVD<Eigen::MatrixX3d> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
        svd.setThreshold(std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(nGood, 3L)));
        Eigen::Vector3d const delta = svd.solve(rhs);
        if (!delta.allFinite()) continue;
        newX[j] = s.x + std::min(std::max(delta[1], -maxShift), maxShift);
        newY[j] = s.y + std::min(std::max(delta[2], -maxShift), maxShift);
        moved[j] = 1;
    }
}

}  // namespace crowd
}  // namespace meas
}  // namespace lsst
