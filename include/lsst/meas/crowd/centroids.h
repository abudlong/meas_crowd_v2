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
#ifndef LSST_MEAS_CROWD_CENTROIDS_H
#define LSST_MEAS_CROWD_CENTROIDS_H

#include <cstdint>
#include <vector>

#include "lsst/meas/crowd/jointFit.h"

namespace lsst {
namespace meas {
namespace crowd {

/// One source to move: position (parent pixel coordinates), flux, and its fitted (tapered) PSF
/// stamp clipped to the image (origin in image-array coordinates).
struct CentroidSource {
    double x = 0, y = 0, amp = 0;
    StampView stamp;
};

/**
 * One Gauss-Newton position step per source (pipeline._refineSubset): on the good pixels within
 * sqrt(R2) of the source, fit the residual with [PSF, amp dPSF/dx, amp dPSF/dy] weighted by
 * 1 / sqrt(var) (minimum-norm least squares, numpy's default rank cut-off), and move by the
 * position terms clipped to +-maxShift.  The derivatives are central differences (step eps) of
 * the Lanczos-shifted kernel over the window [floor(x) - rWin, floor(x) + rWin + 1] (same for y).
 * Images are ny x nx row-major with parent origin (x0Img, y0Img).  Sources that cannot be moved
 * (fewer than 10 pixels, window off the image, stamp mismatch) keep moved[j] = 0.
 */
void refineCentroids(double const *kernel, long kny, long knx, int order, double eps,
                     std::vector<CentroidSource> const &sources, double const *resid,
                     double const *var, std::uint8_t const *good, long ny, long nx, long x0Img,
                     long y0Img, double R2, long rWin, double maxShift, double *newX, double *newY,
                     std::uint8_t *moved);

}  // namespace crowd
}  // namespace meas
}  // namespace lsst

#endif  // LSST_MEAS_CROWD_CENTROIDS_H
