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
INLINE DEVICE std::uint32_t
  shfl_down(T a, I src)
{
  return __shfl_down(a, src);
}

template<typename T>
INLINE DEVICE T
  atomic_add(T* a, T b)
{
  return atomicAdd(a, b);
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

template<typename T>
INLINE DEVICE T
  atomic_add(T* a, T b)
{
  const auto old = *a;
  *a += b;
  return old;
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
template<typename T>
using Vec3 = toolbox::Vec3<T>;

template<typename T, typename F>
INLINE DEVICE T
  warp_reduce(T t, F f)
{
#pragma unroll
  for(auto src_lane = detail::wsz() / 2u; src_lane >= 1; src_lane /= 2u) {
    t = f(t, detail::shfl_down(t, src_lane));
  }

  return t;
}

template<typename value_type>
struct Box {
  using Vec3v = Vec3<value_type>;

  Vec3v min = { std::numeric_limits<value_type>::max() };
  Vec3v max = { std::numeric_limits<value_type>::min() };

  DEVICE bool contains(const Vec3v& point) const
  {
    bool contained = true;
#pragma unroll
    for(auto i = 0u; i < 3u; i++) {
      contained &= min[i] <= point[i] && point[i] < max[i];
    }

    return contained;
  }

  DEVICE Vec3v extent() const { return max - min; }

  DEVICE value_type volume() const
  {
    const auto ext = extent();
    return ext.x * ext.y * ext.z;
  }

  DEVICE value_type volume(const Vec3v& ext) const { return ext.x * ext.y * ext.z; }
};

template<typename value_type>
Box<value_type>
  make_bounding_box(
    std::uint32_t particle_offset,
    const value_type cfl,
    const std::array<value_type, 3> lattice_origo_coordinates,
    std::span<runko::prtc_id_type> ids_span,
    std::span<value_type> vel_span,
    std::span<value_type> pos_span,
    std::span<value_type> scratch)
{
  Box<value_type> box;
  // First we loop over the amount of particles we want to consider for the
  // bounding box. Each thread keeps a tally of the minimum and maximum for each axis.
  for(auto elem_idx = detail::tid(); elem_idx < detail::bdim();
      elem_idx += detail::bdim()) {
    const auto particle_idx = particle_offset + elem_idx;
    if(ids_span[particle_idx] == runko::dead_prtc_id) { continue; }

    const auto u = Vec3<value_type>(vel_span[particle_idx]).template as<value_type>();
    const auto invgam =
      value_type { 1 } / sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));

    const auto x2 =
      Vec3<value_type>(pos_span[particle_idx]).template as<value_type>() -
      Vec3<value_type>(lattice_origo_coordinates).template as<value_type>();

    const auto x1 = x2 - cfl * invgam * u;
    const auto fi1 =
      Vec3<value_type>(sstd::floor(x1(0)), sstd::floor(x1(1)), sstd::floor(x1(2)));
    const auto fi2 =
      Vec3<value_type>(sstd::floor(x2(0)), sstd::floor(x2(1)), sstd::floor(x2(2)));

#pragma unroll
    for(auto i = 0u; i < 3; i++) {
      box.min[i] = std::min(box.min[i], std::min(fi1[i], fi2[i]));
      box.max[i] = std::max(box.max[i], std::max(fi1[i], fi2[i]));
    }
  }

  // Next we do a warp reduction over the bounds: after this, threads on lane == 0
  // will have the bounds from its warp
#pragma unroll
  for(auto i = 0u; i < 3u; i++) {
    box.min[i] = warp_reduce(box.min[i], std::min<value_type>);
    box.max[i] = warp_reduce(box.max[i], std::max<value_type>);
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
    const auto* f = i < 3u ? std::min<value_type> : std::max<value_type>;

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
  box.max += Vec3<value_type> { 2 };
  auto extent = box.extent();
  auto volume = box.volume(extent);

  // If the entire box doesn't fit into the available scratch memory, we must
  // reduce the size of the box. This could be improved, now it just reduces the
  // largest dimension by one, until the box fits. We must multiply the volume by
  // three, as we'll be saving a vector of currents.
  while(scratch.size() < 3u * volume) {
    const auto i = extent[0u] > extent[1u] ? (extent[0u] > extent[2u] ? 0u : 2u)
                                           : (extent[1u] > extent[2u] ? 1u : 2u);
    extent[i] -= value_type { 1 };
    volume = box.volume(extent);
  }

  // The new maximum is minimum + the (possibly) reduced extent.
  // Something else could be done as well, e.g. using the center
  // and computing both min and max.
  box.max = box.min + extent;

  return box;
}

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
  for(auto chunk_idx = detail::bid(); chunk_idx < num_chunks;
      chunk_idx += detail::gdim()) {
    const auto particle_offset = chunk_idx * chunk_size;

    const auto box = make_bounding_box(
      particle_offset,
      cfl,
      lattice_origo_coordinates,
      ids_span,
      vel_span,
      pos_span,
      scratch);

    for(auto i = detail::tid(); i < scratch.size(); i += detail::bdim()) {
      scratch[i] = value_type { 0 };
    }

    detail::syncthreads();

    const auto store_current =
      [&](const Vec3i& point, const Vec3<value_type>& current) {
        if(box.contains(point)) {
          const auto delta  = point - box.min;
          const auto extent = box.extent();
          const auto volume = box.volume(extent);
          const auto idx = delta.z + delta.y * extent.z + delta.x * extent.y * extent.z;
#pragma unroll
          for(auto i = 0u; i < 3u; i++) {
            detail::atomic_add(&scratch[idx + i * volume], current[i]);
          }
        } else {
          const auto si  = point.template as<runko::index_t>();
          auto* const Jx = &thrust::raw_reference_cast(Jmds[si.data][0]);
          auto* const Jy = &thrust::raw_reference_cast(Jmds[si.data][1]);
          auto* const Jz = &thrust::raw_reference_cast(Jmds[si.data][2]);

          detail::atomic_add(Jx, current[0]);
          detail::atomic_add(Jy, current[1]);
          detail::atomic_add(Jz, current[2]);
        }
      };

    for(auto elem_idx = detail::tid(); elem_idx < chunk_size;
        elem_idx += detail::bdim()) {
      const auto particle_idx = particle_offset + elem_idx;
      if(ids_span[particle_idx] == runko::dead_prtc_id) { continue; }

      const auto u = Vec3<value_type>(vel_span[particle_idx]).template as<value_type>();
      const auto invgam =
        value_type { 1 } / sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));

      const auto x2 =
        Vec3<value_type>(pos_span[particle_idx]).template as<value_type>() -
        Vec3<value_type>(lattice_origo_coordinates).template as<value_type>();

      const auto x1 = x2 - cfl * invgam * u;

      // Float floor for relay and weight computation (pure float — no 64-bit
      // integers)
      const auto fi1 =
        Vec3<value_type>(sstd::floor(x1(0)), sstd::floor(x1(1)), sstd::floor(x1(2)));
      const auto fi2 =
        Vec3<value_type>(sstd::floor(x2(0)), sstd::floor(x2(1)), sstd::floor(x2(2)));

      const auto relay = [&](const runko::index_t j) -> value_type {
        const auto a  = sstd::min(fi1(j), fi2(j)) + value_type { 1 };
        const auto b1 = sstd::max(fi1(j), fi2(j));
        const auto b2 = value_type { 0.5 } * (x1(j) + x2(j));
        const auto b  = sstd::max(b1, b2);
        return sstd::min(a, b);
      };

      const auto x_relay = Vec3<value_type>(relay(0), relay(1), relay(2));

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
        Vec3<value_type>(
          Fx1 * (one - Wy1) * (one - Wz1),
          Fy1 * (one - Wx1) * (one - Wz1),
          Fz1 * (one - Wx1) * (one - Wy1)));

      store_current(
        i2,
        Vec3<value_type>(
          Fx2 * (one - Wy2) * (one - Wz2),
          Fy2 * (one - Wx2) * (one - Wz2),
          Fz2 * (one - Wx2) * (one - Wy2)));
      store_current(
        i1 + Vec3i(1, 0, 0),
        Vec3<value_type>(0, Fy1 * Wx1 * (one - Wz1), Fz1 * Wx1 * (one - Wy1)));
      store_current(
        i2 + Vec3i(1, 0, 0),
        Vec3<value_type>(0, Fy2 * Wx2 * (one - Wz2), Fz2 * Wx2 * (one - Wy2)));
      store_current(
        i1 + Vec3i(0, 1, 0),
        Vec3<value_type>(Fx1 * Wy1 * (one - Wz1), 0, Fz1 * (one - Wx1) * Wy1));
      store_current(
        i2 + Vec3i(0, 1, 0),
        Vec3<value_type>(Fx2 * Wy2 * (one - Wz2), 0, Fz2 * (one - Wx2) * Wy2));
      store_current(
        i1 + Vec3i(0, 0, 1),
        Vec3<value_type>(Fx1 * (one - Wy1) * Wz1, Fy1 * (one - Wx1) * Wz1, 0));
      store_current(
        i2 + Vec3i(0, 0, 1),
        Vec3<value_type>(Fx2 * (one - Wy2) * Wz2, Fy2 * (one - Wx2) * Wz2, 0));
      store_current(i1 + Vec3i(0, 1, 1), Vec3<value_type>(Fx1 * Wy1 * Wz1, 0, 0));
      store_current(i2 + Vec3i(0, 1, 1), Vec3<value_type>(Fx2 * Wy2 * Wz2, 0, 0));
      store_current(i1 + Vec3i(1, 0, 1), Vec3<value_type>(0, Fy1 * Wx1 * Wz1, 0));
      store_current(i2 + Vec3i(1, 0, 1), Vec3<value_type>(0, Fy2 * Wx2 * Wz2, 0));
      store_current(i1 + Vec3i(1, 1, 0), Vec3<value_type>(0, 0, Fz1 * Wx1 * Wy1));
      store_current(i2 + Vec3i(1, 1, 0), Vec3<value_type>(0, 0, Fz2 * Wx2 * Wy2));
    }

    detail::syncthreads();

    const auto extent = box.extent();
    const auto volume = box.volume(extent);
    for(auto shmem_idx = detail::tid(); shmem_idx < volume;
        shmem_idx += detail::bdim()) {
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
        const auto val = scratch[shmem_idx + i * volume];
        if(0 != val) { detail::atomic_add(J[i], val); }
      }
    }

    detail::syncthreads();
  }
}
}  // namespace pic
