// Copyright 2025 - 2026, Miro Palmu, Joonas Nättilä and the runko contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "runko/emf/yee_lattice.h"
#include "runko/pic/particle.h"
#include "runko/tools/math.h"
#include "runko/tools/vector.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/iterator/transform_output_iterator.h"
#include "thrust/memory.h"
#include "thrust/reduce.h"
#include "thrust/sort.h"
#include "tyvi/mdgrid.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if defined(TYVI_BACKEND_HIP)
  #include "hip/hip_runtime.h"
  #define GLOBAL __global__
  #define DEVICE __device__
  #define SHARED __shared__
  #define INLINE __forceinline__
#else
  #define GLOBAL
  #define DEVICE
  #define SHARED
  #define INLINE inline
#endif

namespace detail {

#if defined(TYVI_BACKEND_HIP)
INLINE DEVICE void
  syncthreads()
{
  __syncthreads();
}

template<typename T, typename I>
INLINE DEVICE T
  shfl_down(T a, I src)
{
  return __shfl_down(a, src);
}

INLINE DEVICE std::uint32_t
  gdim()
{
  return gridDim.x;
}

INLINE DEVICE std::uint32_t
  bdim()
{
  return blockDim.x;
}

INLINE DEVICE std::uint32_t
  wsz()
{
  return warpSize;
}

INLINE DEVICE std::uint32_t
  num_warps()
{
  return bdim() / wsz();
}

INLINE DEVICE std::uint32_t
  bid()
{
  return blockIdx.x;
}

INLINE DEVICE std::uint32_t
  tid()
{
  return threadIdx.x;
}

INLINE DEVICE std::uint32_t
  wid()
{
  return tid() / wsz();
}

INLINE DEVICE std::uint32_t
  lid()
{
  return tid() & (wsz() - 1u);
}
#else
INLINE DEVICE void
  syncthreads()
{
  return;
}

template<typename T, typename I>
INLINE DEVICE std::uint32_t
  shfl_down(T a, I)
{
  return a;
}

INLINE DEVICE std::uint32_t
  gdim()
{
  return 1u;
}

INLINE DEVICE std::uint32_t
  bdim()
{
  return 1u;
}

INLINE DEVICE std::uint32_t
  wsz()
{
  return 1u;
}

INLINE DEVICE std::uint32_t
  num_warps()
{
  return bdim() / wsz();
}

INLINE DEVICE std::uint32_t
  bid()
{
  return 0u;
}

INLINE DEVICE std::uint32_t
  tid()
{
  return 0u;
}

INLINE DEVICE std::uint32_t
  wid()
{
  return tid() / wsz();
}

INLINE DEVICE std::uint32_t
  lid()
{
  return tid() & (wsz() - 1u);
}
#endif
}  // namespace detail

namespace pic {
using Vec3i = toolbox::Vec3<std::uint32_t>;

template<typename T, typename F>
INLINE DEVICE T
  warp_reduce(T t, F f)
{
#pragma unroll
  for(auto src_lane = detail::wsz() / 2u; src_lane >= 1u; src_lane /= 2u) {
    t = f(t, detail::shfl_down(t, src_lane));
  }

  return t;
}

// This assumes cells can never be negative
struct Box {
  Vec3i min = { ~0u };
  Vec3i max = { 0u };

  DEVICE INLINE bool contains(const Vec3i& point) const
  {
    bool contained = true;
#pragma unroll
    for(auto i = 0u; i < 3u; i++) {
      contained &= min[i] <= point[i] && point[i] < max[i];
    }

    return contained;
  }

  DEVICE INLINE Vec3i extent() const { return max - min; }
};

std::uint32_t
  volume(const Vec3i& v)
{
  return v.x * v.y * v.z;
}

template<typename value_type>
DEVICE INLINE Box
  make_bounding_box(
    const value_type cfl,
    const std::array<value_type, 3> lattice_origo_coordinates,
    std::span<runko::prtc_id_type> ids_span,
    std::span<value_type> vel_span,
    std::span<value_type> pos_span,
    std::span<value_type> scratch)
{
  using Vec3v = toolbox::Vec3<value_type>;
  Box box;
  // First we loop over the amount of particles we want to consider for the
  // bounding box. Each thread keeps a tally of the minimum and maximum for each axis.
  for(auto elem_idx = detail::tid(); elem_idx < ids_span.size();
      elem_idx += detail::bdim()) {
    if(ids_span[elem_idx] == runko::dead_prtc_id) { continue; }

    const auto u = Vec3v(vel_span[elem_idx]);
    const auto invgam =
      value_type { 1 } / sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));
    const auto x2 = Vec3v(pos_span[elem_idx]) - Vec3v(lattice_origo_coordinates);
    const auto x1 = x2 - cfl * invgam * u;
    const auto i1 = x1.template as<std::uint32_t>();
    const auto i2 = x2.template as<std::uint32_t>();

#pragma unroll
    for(auto i = 0u; i < 3; i++) {
      box.min[i] = sstd::min(box.min[i], sstd::min(i1[i], i2[i]));
      box.max[i] = sstd::max(box.max[i], sstd::max(i1[i], i2[i]));
    }
  }

  // Next we do a warp reduction over the bounds: after this, threads with lane id == 0
  // will have the bounds from its warp
#pragma unroll
  for(auto i = 0u; i < 3u; i++) {
    box.min[i] = warp_reduce(box.min[i], sstd::min<std::uint32_t>);
    box.max[i] = warp_reduce(box.max[i], sstd::max<std::uint32_t>);
  }

  // If you're lane 0, store the six different values to scratch memory
  // (scratch == LDS/shared memory for GPUs)
  // If we have, say 16 warps, scratch will contain the 16 min[0] values,
  // then the 16 min[1] values, then 16 min[2] values, then 16 max[0] values
  // and so on.
  if(0 == detail::lid()) {
#pragma unroll
    for(auto i = 0u; i < 3u; i++) {
      scratch[detail::wid() + i * detail::num_warps()]        = box.min[i];
      scratch[detail::wid() + (3u + i) * detail::num_warps()] = box.max[i];
    }
  }

  detail::syncthreads();

  // We have six different bounds in the scratch memory, the bounds from this block
  // Six different warps will reduces the six different values.
  for(auto i = detail::wid(); i < 6u; i += detail::num_warps()) {
    // All threads participate to avoid deadlocks, in case
    // we are using a small block size (<= 6 * 64),
    // in which case this for loop is necessary.
    value_type val = { 0 };
    if(detail::lid() < detail::num_warps()) {
      val = scratch[detail::lid() + i * detail::num_warps()];
    }
    // First three warps do the minimum bounds, the other three max
    const auto* f = i < 3u ? sstd::min<value_type> : sstd::max<value_type>;

    // Not a warp-wide reduction, only however many warps we have in the block
    // (e.g. 16 for a block of 1024 threads)
    for(auto j = detail::num_warps() / 2u; j >= 1u; j /= 2u) {
      val = f(val, detail::shfl_down(val, j));
    }

    // Store back in scratch, but make sure to write in the area used by your
    // own warp, to avoid data races with other warps.
    if(0 == detail::lid()) { scratch[i * detail::num_warps()] = val; }
  }

  detail::syncthreads();

  // After this, every thread in the block has the block-wide bounds
#pragma unroll
  for(auto i = 0u; i < 3u; i++) {
    box.min[i] = scratch[i * detail::num_warps()];
    box.max[i] = scratch[(i + 3u) * detail::num_warps()];
  }

  // We'll add two to the bounds:
  // 1 because we want the upper bound be higher than any value inside the box
  // (if all particles are in the same cell, upper and lower bounds are the
  // same after the reduction)
  // and another 1, because we'll be adding current to neighbour cells in the
  // positive directions. Thus, the minimum possible size for the box is
  // (2, 2, 2), if all particles are in the same cell.
  box.max += Vec3v { 2 };
  auto extent = box.extent();
  auto vol    = volume(extent);

  // If the entire box doesn't fit into the available scratch memory, we must
  // reduce the size of the box. This could be improved, now it just reduces the
  // largest dimension by one, until the box fits. We must multiply the volume by
  // three, as we'll be saving a vector of currents.
  while(scratch.size() < 3u * vol) {
    const auto i = extent[0u] > extent[1u] ? (extent[0u] > extent[2u] ? 0u : 2u)
                                           : (extent[1u] > extent[2u] ? 1u : 2u);
    extent[i] -= 1u;
    vol = volume(extent);
  }

  // The new maximum is minimum + the (possibly) reduced extent.
  // Something else could be done as well, e.g. using the center
  // and computing both min and max.
  box.max = box.min + extent;

  return box;
}

// TODO instead of the chunk loop here,
// actually use spans.
// Maybe the chunks can be computed on the host?
// Then we just have e.g. spans of spans. Dunno.
// Anyway, this function can be turned into a kernel,
// then have another function which does what this does inside
// the first loop. That one takes spans, which we create from the chunk sizes
// here
template<typename value_type, typename MDS>
DEVICE void
  deposit_current(
    std::uint32_t num_chunks,
    std::uint32_t chunk_size,
    const value_type cfl,
    const value_type charge,
    std::span<runko::prtc_id_type> ids_span,
    std::span<value_type> vel_span,
    std::span<value_type> pos_span,
    MDS Jmds,
    const std::array<value_type, 3> lattice_origo_coordinates,
    std::span<value_type> scratch)
{
  // 2x2x2 box of Vec3 is the absolute minimum.
  // It won't be efficient, but it's possible.
  assert(scratch.size() >= 24ul);

  using Vec3v = toolbox::Vec3<value_type>;
  for(auto chunk_idx = detail::bid(); chunk_idx < num_chunks;
      chunk_idx += detail::gdim()) {
    const auto particle_offset = chunk_idx * chunk_size;

    // Make subspans for the box size computation.
    const auto count = sstd::min(detail::bdim(), chunk_size);
    const auto box   = make_bounding_box(
      cfl,
      lattice_origo_coordinates,
      ids_span.subspan(particle_offset, count),
      pos_span.subspan(particle_offset, count),
      vel_span.subspan(particle_offset, count),
      scratch);

    for(auto i = detail::tid(); i < scratch.size(); i += detail::bdim()) {
      scratch[i] = value_type { 0 };
    }

    detail::syncthreads();

    const auto extent        = box.extent();
    const auto vol           = volume(extent);
    const auto store_current = [&](const Vec3i& point, const Vec3v& current) {
      if(box.contains(point)) {
        const auto delta = point - box.min;
        const auto idx   = delta.z + delta.y * extent.z + delta.x * extent.y * extent.z;
#pragma unroll
        for(auto i = 0u; i < 3u; i++) {
          sstd::atomic_add(&scratch[idx + i * vol], current[i]);
        }
      } else {
        const auto si  = point.template as<runko::index_t>();
        auto* const Jx = &thrust::raw_reference_cast(Jmds[si.data][0]);
        auto* const Jy = &thrust::raw_reference_cast(Jmds[si.data][1]);
        auto* const Jz = &thrust::raw_reference_cast(Jmds[si.data][2]);

        sstd::atomic_add(Jx, current[0]);
        sstd::atomic_add(Jy, current[1]);
        sstd::atomic_add(Jz, current[2]);
      }
    };

    for(auto elem_idx = detail::tid(); elem_idx < chunk_size;
        elem_idx += detail::bdim()) {
      const auto particle_idx = particle_offset + elem_idx;
      if(ids_span[particle_idx] == runko::dead_prtc_id) { continue; }

      const auto u = Vec3v(vel_span[particle_idx]).template as<value_type>();
      const auto invgam =
        value_type { 1 } / sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));

      const auto x2 = Vec3v(pos_span[particle_idx]).template as<value_type>() -
                      Vec3v(lattice_origo_coordinates).template as<value_type>();

      const auto x1 = x2 - cfl * invgam * u;

      // Float floor for relay and weight computation (pure float — no 64-bit
      // integers)
      const auto fi1 =
        Vec3v(sstd::floor(x1(0)), sstd::floor(x1(1)), sstd::floor(x1(2)));
      const auto fi2 =
        Vec3v(sstd::floor(x2(0)), sstd::floor(x2(1)), sstd::floor(x2(2)));

      const auto relay = [&](const runko::index_t j) -> value_type {
        const auto a  = sstd::min(fi1(j), fi2(j)) + value_type { 1 };
        const auto b1 = sstd::max(fi1(j), fi2(j));
        const auto b2 = value_type { 0.5 } * (x1(j) + x2(j));
        const auto b  = sstd::max(b1, b2);
        return sstd::min(a, b);
      };

      const auto x_relay = Vec3v(relay(0), relay(1), relay(2));

      const auto F1 = charge * (x_relay - x1);
      const auto F2 = charge * (x2 - x_relay);

      const auto i1 = fi1.template as<uint32_t>();
      const auto i2 = fi2.template as<uint32_t>();

      const auto W1 = value_type { 0.5 } * (x1 + x_relay) - fi1;
      const auto W2 = value_type { 0.5 } * (x2 + x_relay) - fi2;

      const auto [Fx1, Fy1, Fz1] = F1.data;
      const auto [Fx2, Fy2, Fz2] = F2.data;
      const auto [Wx1, Wy1, Wz1] = W1.data;
      const auto [Wx2, Wy2, Wz2] = W2.data;

      static constexpr auto one = value_type { 1 };

      store_current(
        i1,
        Vec3v(
          Fx1 * (one - Wy1) * (one - Wz1),
          Fy1 * (one - Wx1) * (one - Wz1),
          Fz1 * (one - Wx1) * (one - Wy1)));

      store_current(
        i2,
        Vec3v(
          Fx2 * (one - Wy2) * (one - Wz2),
          Fy2 * (one - Wx2) * (one - Wz2),
          Fz2 * (one - Wx2) * (one - Wy2)));
      store_current(
        i1 + Vec3i(1, 0, 0),
        Vec3v(0, Fy1 * Wx1 * (one - Wz1), Fz1 * Wx1 * (one - Wy1)));
      store_current(
        i2 + Vec3i(1, 0, 0),
        Vec3v(0, Fy2 * Wx2 * (one - Wz2), Fz2 * Wx2 * (one - Wy2)));
      store_current(
        i1 + Vec3i(0, 1, 0),
        Vec3v(Fx1 * Wy1 * (one - Wz1), 0, Fz1 * (one - Wx1) * Wy1));
      store_current(
        i2 + Vec3i(0, 1, 0),
        Vec3v(Fx2 * Wy2 * (one - Wz2), 0, Fz2 * (one - Wx2) * Wy2));
      store_current(
        i1 + Vec3i(0, 0, 1),
        Vec3v(Fx1 * (one - Wy1) * Wz1, Fy1 * (one - Wx1) * Wz1, 0));
      store_current(
        i2 + Vec3i(0, 0, 1),
        Vec3v(Fx2 * (one - Wy2) * Wz2, Fy2 * (one - Wx2) * Wz2, 0));
      store_current(i1 + Vec3i(0, 1, 1), Vec3v(Fx1 * Wy1 * Wz1, 0, 0));
      store_current(i2 + Vec3i(0, 1, 1), Vec3v(Fx2 * Wy2 * Wz2, 0, 0));
      store_current(i1 + Vec3i(1, 0, 1), Vec3v(0, Fy1 * Wx1 * Wz1, 0));
      store_current(i2 + Vec3i(1, 0, 1), Vec3v(0, Fy2 * Wx2 * Wz2, 0));
      store_current(i1 + Vec3i(1, 1, 0), Vec3v(0, 0, Fz1 * Wx1 * Wy1));
      store_current(i2 + Vec3i(1, 1, 0), Vec3v(0, 0, Fz2 * Wx2 * Wy2));
    }

    detail::syncthreads();

    for(auto shmem_idx = detail::tid(); shmem_idx < vol; shmem_idx += detail::bdim()) {
      const auto si = (Vec3i { shmem_idx / (extent.y * extent.z),
                               (shmem_idx / extent.z) % extent.y,
                               shmem_idx % extent.z } +
                       box.min)
                        .template as<runko::index_t>();
      const auto* const J[3] = {
        &thrust::raw_reference_cast(Jmds[si.data][0]),
        &thrust::raw_reference_cast(Jmds[si.data][1]),
        &thrust::raw_reference_cast(Jmds[si.data][2]),
      };

#pragma unroll
      for(auto i = 0u; i < 3u; i++) {
        const auto val = scratch[shmem_idx + i * vol];
        if(0 != val) { sstd::atomic_add(J[i], val); }
      }
    }

    detail::syncthreads();
  }
}
}  // namespace pic
