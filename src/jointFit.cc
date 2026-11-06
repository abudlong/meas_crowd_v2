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
// Per-source loops of the joint fit (fitLib.jointPsfFitDiag) in C++: the design matrix with its
// source groups, and the model image with the group map.  Same arithmetic, in the same order, as
// the Python they replace, so the results are identical.
#include "lsst/meas/crowd/jointFit.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace lsst {
namespace meas {
namespace crowd {

namespace {

long findRoot(std::vector<long> &parent, long i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

void unite(std::vector<long> &parent, long a, long b) {
    a = findRoot(parent, a);
    b = findRoot(parent, b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
}

double stampMax(StampView const &s) {
    double m = -std::numeric_limits<double>::infinity();
    for (long r = 0; r < s.ny; ++r) {
        double const *row = s.data + r * s.rowStride;
        for (long c = 0; c < s.nx; ++c) m = std::max(m, row[c * s.colStride]);
    }
    return m;
}

}  // namespace

DesignMatrix buildDesignMatrix(std::vector<StampView> const &stamps, std::int64_t const *pixelIndex,
                               long ny, long nx, double const *sqrtW, long nPix, double supportFrac) {
    long const n = static_cast<long>(stamps.size());
    DesignMatrix out;
    out.indptr.assign(n + 1, 0);
    out.sigOffset.assign(n + 1, 0);
    std::vector<long> parent(n);
    std::iota(parent.begin(), parent.end(), 0L);
    std::size_t total = 0;                      // upper bound on the entries: reserve once
    for (StampView const &s : stamps)
        if (s.data != nullptr) total += static_cast<std::size_t>(s.ny * s.nx);
    out.indices.reserve(total);
    out.data.reserve(total);
    out.sig.reserve(total);
    // the first source with a significant good pixel there, per good pixel (for the groups)
    std::vector<long> owner(static_cast<std::size_t>(std::max(nPix, 0L)), -1);
    for (long j = 0; j < n; ++j) {
        StampView const &s = stamps[j];
        out.sigOffset[j + 1] = out.sigOffset[j];
        if (s.data == nullptr) {
            out.indptr[j + 1] = static_cast<std::int32_t>(out.indices.size());
            continue;
        }
        if (s.x0 < 0 || s.y0 < 0 || s.x0 + s.nx > nx || s.y0 + s.ny > ny)
            throw std::invalid_argument("stamp outside the image");
        double const smax = stampMax(s);
        double const thr = supportFrac * smax;
        bool const anySig = smax > 0;
        for (long r = 0; r < s.ny; ++r) {
            double const *row = s.data + r * s.rowStride;
            std::int64_t const *pix = pixelIndex + (s.y0 + r) * nx + s.x0;
            for (long c = 0; c < s.nx; ++c) {
                double const v = row[c * s.colStride];
                bool const sg = anySig && v > thr;
                out.sig.push_back(sg ? 1 : 0);
                std::int64_t const p = pix[c];
                if (p < 0 || v == 0.0) continue;
                if (p >= nPix) throw std::out_of_range("pixel index beyond sqrtW");
                out.indices.push_back(static_cast<std::int32_t>(p));
                out.data.push_back(v * sqrtW[p]);
                if (sg) {
                    if (owner[p] < 0) {
                        owner[p] = j;
                    } else {
                        unite(parent, owner[p], j);
                    }
                }
            }
        }
        out.sigOffset[j + 1] = static_cast<std::int64_t>(out.sig.size());
        if (out.indices.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::length_error("design matrix too large for 32-bit indices");
        out.indptr[j + 1] = static_cast<std::int32_t>(out.indices.size());
    }
    // number the groups in order of their lowest source index (as scipy's connected_components)
    out.labels.assign(n, -1);
    std::vector<std::int32_t> rootLabel(n, -1);
    int next = 0;
    for (long j = 0; j < n; ++j) {
        long const r = findRoot(parent, j);
        if (rootLabel[r] < 0) rootLabel[r] = next++;
        out.labels[j] = rootLabel[r];
    }
    out.nGroups = next;
    return out;
}

void accumulateModel(std::vector<StampView> const &stamps, double const *amps, double ampZeroTol,
                     std::int32_t const *labels, std::uint8_t const *sig,
                     std::int64_t const *sigOffset, long ny, long nx, double *model,
                     std::int64_t *groupMap) {
    long const n = static_cast<long>(stamps.size());
    for (long j = 0; j < n; ++j) {  // model, sources in order (same additions as the Python loop)
        StampView const &s = stamps[j];
        if (s.data == nullptr || !(amps[j] > ampZeroTol)) continue;
        double const a = amps[j];
        for (long r = 0; r < s.ny; ++r) {
            double const *row = s.data + r * s.rowStride;
            double *m = model + (s.y0 + r) * nx + s.x0;
            for (long c = 0; c < s.nx; ++c) m[c] += a * row[c * s.colStride];
        }
    }
    for (long j = 0; j < n; ++j) {  // group map: later sources overwrite earlier ones
        StampView const &s = stamps[j];
        if (s.data == nullptr) continue;
        std::uint8_t const *sg = sig + sigOffset[j];
        for (long r = 0; r < s.ny; ++r) {
            std::int64_t *g = groupMap + (s.y0 + r) * nx + s.x0;
            for (long c = 0; c < s.nx; ++c)
                if (sg[r * s.nx + c]) g[c] = labels[j];
        }
    }
}

}  // namespace crowd
}  // namespace meas
}  // namespace lsst
