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
#ifndef LSST_MEAS_CROWD_JOINTFIT_H
#define LSST_MEAS_CROWD_JOINTFIT_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lsst {
namespace meas {
namespace crowd {

/// One source's PSF stamp, already clipped to the image: values row-major with the given
/// strides (in elements), origin (x0, y0) in image-array coordinates.  data == nullptr: no stamp.
struct StampView {
    double const *data = nullptr;
    long ny = 0, nx = 0;
    long rowStride = 0, colStride = 1;
    long x0 = 0, y0 = 0;
};

/// What `buildDesignMatrix` returns.
struct DesignMatrix {
    std::vector<std::int32_t> indptr;   ///< CSC column pointers, nSources + 1
    std::vector<std::int32_t> indices;  ///< row (good-pixel index) of every entry
    std::vector<double> data;           ///< stamp value x sqrt(weight) of every entry
    std::vector<std::int32_t> labels;   ///< group of every source (sources sharing significant pixels)
    int nGroups = 0;
    std::vector<std::uint8_t> sig;      ///< significant-support mask of every stamp, concatenated
    std::vector<std::int64_t> sigOffset;  ///< start of every stamp's mask in `sig`, nSources + 1
};

/**
 * Build the weighted joint-fit design matrix (as `fitLib.jointPsfFitDiag` does in Python).
 *
 * For every source j and stamp pixel with a good-pixel index p >= 0 and a non-zero value s:
 * entry (p, j) = s * sqrtW[p].  A pixel is significant when s > supportFrac * max(stamp); sources
 * sharing a significant good pixel are in the same group, numbered by their lowest source index
 * (the numbering of scipy's connected_components).
 *
 * @param stamps      one view per source
 * @param pixelIndex  good-pixel index per image pixel (-1 = not used), ny x nx row-major
 * @param ny, nx      image shape
 * @param sqrtW       sqrt of the weight of every good pixel (nPix values)
 * @param supportFrac significance threshold, fraction of each stamp's maximum
 */
DesignMatrix buildDesignMatrix(std::vector<StampView> const &stamps, std::int64_t const *pixelIndex,
                               long ny, long nx, double const *sqrtW, long nPix, double supportFrac);

/**
 * Model image and group map of a joint fit: model += amps[j] * stamp_j for amps[j] > ampZeroTol
 * (sources in order), and groupMap = labels[j] on stamp j's significant pixels (later sources
 * overwrite earlier ones), as `fitLib.jointPsfFitDiag` does in Python.
 */
void accumulateModel(std::vector<StampView> const &stamps, double const *amps, double ampZeroTol,
                     std::int32_t const *labels, std::uint8_t const *sig,
                     std::int64_t const *sigOffset, long ny, long nx, double *model,
                     std::int64_t *groupMap);

}  // namespace crowd
}  // namespace meas
}  // namespace lsst

#endif  // LSST_MEAS_CROWD_JOINTFIT_H
