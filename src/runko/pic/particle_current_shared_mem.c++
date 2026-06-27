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
#include <cstddef>
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

// Claude, here's some guides for you.
// Assumptions:
// - value_type = float
// - runko::dead_prtc_id == maximum value of std::uint64_t
//
// TODO
// - pos values are always positive and they start from
// - a = 3 and
//
// write the following tests:
// - for GPU (assume MI250X):
// 1. Compare detail::shfl_down to the intrinsic __shfl_down
// 2. Test detail::num_warps for different block sizes
// 3. Test detail::wid for different block sizes, always less than num_warps
// 4. Test detail::lid is always below detail::wsz and values are different
// 5. Test that make_aligned_span returns a span, the pointer of which is aligned
//    correctly for that type. Test multiple types.
// 6. Test that warp_reduce correctly reduces for different functions
//    test multiple binary functions: min, max, add, sub, mul, div and so on
// 7. Box:
//    Assume all points coordinates are non-negative
//    7.1: Create a box with some min and max and check that `contains`
//        works correctly for a bunch of points.
//    7.2: create a box with some min and max and check that `extent` is correct
//    7.3: `bound_points` should increase the size of the box such that it'll contain
//         all the points that `bound_points` was called with and any other points
//         that have coordinate values within the limits of the maximum and minimum
//         coordinate values of all the points.
//    7.4: Given a scratch (LDS) that fits at least 96 std::uint32_t values,
//         check `bound_thread_boxes`. Create a unique box for each thread.
//         After `bound_thread_boxes` all threads should have the same box:
//         one box that bounds all the other boxes.
//    7.5: With max_volume in range [24, 64000/2/8/3] check that `shrink_to_max_volume`
//         shrinks boxes of different sizes (and shapes!) such that the volume of the
//         box multiplied by three is then less than the max_volume.
// 9. Check prod computes the product of a vector correctly

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

template<typename T>
INLINE DEVICE std::span<T>
  make_aligned_span(std::span<std::byte> byte_span)
{
  static constexpr auto alignment = std::alignment_of_v<T>;
  const auto byte_address         = reinterpret_cast<std::uintptr_t>(byte_span.data());
  const auto bytes_over_alignment = byte_address & (alignment - 1ul);
  const auto padding =
    bytes_over_alignment > 0ul ? alignment - bytes_over_alignment : 0ul;
  return std::span<T>(byte_span.data() + padding, byte_span.size() - padding);
}

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

template<typename T>
T
  prod(const toolbox::Vec3<T>& v)
{
  return v.x * v.y * v.z;
}

// This assumes cells can never be negative
template<typename value_type>
struct Box {
  Vec3i min                             = { ~0u };
  Vec3i max                             = { 0u };
  std::span<value_type> current_scratch = {};

  DEVICE INLINE Box(std::span<std::byte> scratch) :
    current_scratch(make_aligned_span<value_type>(scratch))
  {
  }

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

  DEVICE INLINE void bound_points(const Vec3i& i1, const Vec3i& i2)
  {
#pragma unroll
    for(auto i = 0u; i < 3; i++) {
      min[i] = sstd::min(min[i], sstd::min(i1[i], i2[i]));
      max[i] = sstd::max(max[i], sstd::max(i1[i], i2[i]));
    }
  }

  DEVICE INLINE void bound_thread_boxes(std::span<std::uint32_t> scratch)
  {
    // Each thread has their own bounding box. This function
    // bounds all the individual bounding boxes to a block-wide bounding box,
    // which each of the threads will then share.

    // Gather the bounds from the lanes of the warp to lane 0
#pragma unroll
    for(auto i = 0u; i < 3u; i++) {
      min[i] = warp_reduce(min[i], sstd::min<std::uint32_t>);
      max[i] = warp_reduce(max[i], sstd::max<std::uint32_t>);
    }

    // Lane 0 stores the bounds to scratch memory:
    // First num_warps values contain min[0] for each warp,
    // next num_warps values contain min[1] for each warp,
    // then min[2],
    // then max[0] and so on.
    if(0 == detail::lid()) {
#pragma unroll
      for(auto i = 0u; i < 3u; i++) {
        scratch[detail::wid() + i * detail::num_warps()]        = min[i];
        scratch[detail::wid() + (3u + i) * detail::num_warps()] = max[i];
      }
    }

    detail::syncthreads();

    // If num_warps is less than six (i.e. blockDim.x < 6 * 64), some warps need
    // to reduce more than one bound. If there are more than six warps, the first
    // six warps do the reductions is parallel.
    for(auto i = detail::wid(); i < 6u; i += detail::num_warps()) {
      // Each thread participates to avoid deadlock with syncthreads
      std::uint32_t val = 0u;

      // Only lanes with lane id < num_warps read the num_warps values from scratch
      if(detail::lid() < detail::num_warps()) {
        val = scratch[detail::lid() + i * detail::num_warps()];
      }
      const auto* f = i < 3u ? sstd::min<std::uint32_t> : sstd::max<std::uint32_t>;

      // All lanes participate, but lane 0 only contains reductions from first num_warps
      // lanes.
      for(auto j = detail::num_warps() / 2u; j >= 1u; j /= 2u) {
        val = f(val, detail::shfl_down(val, j));
      }

      // Lane 0 stores the reduced bound back to scratch.
      // Store in the same location this warp read from to avoid data races with
      // other warps.
      if(0 == detail::lid()) { scratch[i * detail::num_warps()] = val; }
    }

    detail::syncthreads();

    // Finally, every thread of every warp reads the block-global bounds
    // from scratch memory. After this, every thread has the box that bounds
    // all the points considered by the block.
#pragma unroll
    for(auto i = 0u; i < 3u; i++) {
      min[i] = scratch[i * detail::num_warps()];
      max[i] = scratch[(i + 3u) * detail::num_warps()];
    }
  }

  DEVICE INLINE void shrink_to_max_volume(std::uint32_t max_volume)
  {
    // The box is large enough to bound all the particles.
    // It may be too large to fit into scratch memory, so it may have to
    // be shrunk. We do that here.

    // Increase size by two:
    // If all the particles are in the same cell, the min and max bounds are the same.
    // To make the upper bound strictly larger than any of the particle positions,
    // increase it by one.
    // The other one comes from the stencil operation to the J grid:
    // we'll be adding current to the neighbouring cells in the positive directions,
    // so we'll increase the size in each dimension by one.
    max += Vec3i { 2u };
    auto ext    = extent();
    auto volume = prod(ext);

    // We'll be storing three boxes in the same volume, because we'll be storing
    // a current of Vec3 in the same scratch memory.
    while(max_volume < 3u * volume) {
      const auto i = ext[0u] > ext[1u] ? (ext[0u] > ext[2u] ? 0u : 2u)
                                       : (ext[1u] > ext[2u] ? 1u : 2u);
      ext[i] -= 1u;
      volume = prod(ext);
    }

    max = min + ext;
    // Resize the span
    current_scratch = current_scratch.subspan(0u, 3u * volume);
  }

  DEVICE INLINE void clear_current()
  {
    for(auto i = detail::tid(); i < current_scratch.size(); i += detail::bdim()) {
      current_scratch[i] = value_type { 0 };
    }

    detail::syncthreads();
  }

  DEVICE INLINE void store(const Vec3i& point, const toolbox::Vec3<value_type>& current)
  {
    // TODO: maybe box should store min and extent
    const auto ext    = extent();
    const auto volume = prod(ext);
    const auto delta  = point - min;
    const auto idx    = delta.z + delta.y * ext.z + delta.x * ext.y * ext.z;
#pragma unroll
    for(auto i = 0u; i < 3u; i++) {
      sstd::atomic_add(&current_scratch[idx + i * volume], current[i]);
    }
  }

  template<typename MDS>
  DEVICE INLINE void copy_from_shared_to_global(MDS Jmds)
  {
    // TODO: maybe box should store min and extent
    const auto ext    = extent();
    const auto volume = prod(ext);
    for(auto shmem_idx = detail::tid(); shmem_idx < volume;
        shmem_idx += detail::bdim()) {
      const auto si = (Vec3i { shmem_idx / (extent.y * extent.z),
                               (shmem_idx / extent.z) % extent.y,
                               shmem_idx % extent.z } +
                       min)
                        .template as<runko::index_t>();
      const auto* const J[3] = {
        &thrust::raw_reference_cast(Jmds[si.data][0]),
        &thrust::raw_reference_cast(Jmds[si.data][1]),
        &thrust::raw_reference_cast(Jmds[si.data][2]),
      };

#pragma unroll
      for(auto i = 0u; i < 3u; i++) {
        const auto val = current_scratch[shmem_idx + i * volume];
        if(0 != val) { sstd::atomic_add(J[i], val); }
      }
    }

    detail::syncthreads();
  }
};

template<typename value_type>
DEVICE INLINE Box
  make_bounding_box(
    const value_type cfl,
    std::span<runko::prtc_id_type> ids_span,
    std::span<value_type> vel_span,
    std::span<value_type> pos_span,
    const std::array<value_type, 3> lattice_origo_coordinates,
    std::span<std::byte> scratch)
{
  using Vec3v = toolbox::Vec3<value_type>;
  Box box(scratch);

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

    box.bound_points(i1, i2);
  }

  // Note: we're using the same scratch memory for this, as we'll be using for current
  // later.
  std::span<std::uint32_t> uint_scratch = make_aligned_span<std::uint32_t>(scratch);
  box.bound_thread_boxes(uint_scratch);
  box.shrink_to_max_volume(max_volume);
  box.clear_current();

  return box;
}

template<typename value_type, typename MDS>
DEVICE void
  deposit_current(
    std::uint32_t num_box_candidates,
    const value_type cfl,
    const value_type charge,
    std::span<runko::prtc_id_type> ids_span,
    std::span<value_type> vel_span,
    std::span<value_type> pos_span,
    MDS Jmds,
    const std::array<value_type, 3> lattice_origo_coordinates,
    std::span<std::byte> scratch)
{
  using Vec3v = toolbox::Vec3<value_type>;

  // Make subspans for the box size computation.
  const auto count = sstd::min(ids_span.size(), num_box_candidates);
  const auto box   = make_bounding_box(
    cfl,
    ids_span.subspan(0u, count),
    vel_span.subspan(0u, count),
    pos_span.subspan(0u, count),
    lattice_origo_coordinates,
    scratch);

  const auto store_current = [&](const Vec3i& point, const Vec3v& current) {
    if(box.contains(point)) {
      box.store(point, current);
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

  for(auto elem_idx = detail::tid(); elem_idx < ids_span.size();
      elem_idx += detail::bdim()) {
    if(ids_span[elem_idx] == runko::dead_prtc_id) { continue; }

    const auto u = Vec3v(vel_span[elem_idx]).template as<value_type>();
    const auto invgam =
      value_type { 1 } / sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));

    const auto x2 = Vec3v(pos_span[elem_idx]).template as<value_type>() -
                    Vec3v(lattice_origo_coordinates).template as<value_type>();

    const auto x1 = x2 - cfl * invgam * u;

    // Float floor for relay and weight computation (pure float — no 64-bit
    // integers)
    const auto fi1 = Vec3v(sstd::floor(x1(0)), sstd::floor(x1(1)), sstd::floor(x1(2)));
    const auto fi2 = Vec3v(sstd::floor(x2(0)), sstd::floor(x2(1)), sstd::floor(x2(2)));

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

  box.copy_from_shared_to_global(Jmds);
}

template<typename value_type, typename MDS>
GLOBAL void
  deposit_current_kernel(
    std::uint32_t num_chunks,
    std::uint32_t chunk_size,
    std::uint32_t num_shared_bytes,
    std::uint32_t num_box_candidates,
    const value_type cfl,
    const value_type charge,
    std::span<runko::prtc_id_type> ids_span,
    std::span<value_type> vel_span,
    std::span<value_type> pos_span,
    MDS Jmds,
    const std::array<value_type, 3> lattice_origo_coordinates)
{
  SHARED std::byte scratch[];

  // At least a 2x2x2 box of Vec3 must fit into scratch
  // box creation also uses scratch and requires 6 * num_warps * sizeof(std::uint32_t)
  // bytes
  assert(num_shared_bytes > 3u * 2u * 2u * 2u * sizeof(value_type));
  assert(num_shared_bytes > 6u * detail::num_warps() * sizeof(std::uint32_t));

  std::span<std::byte> scratch_span(scratch, num_shared_bytes);

  // Work over chunks: each block goes over a chunk and chunk size may
  // be different from block dimension. There's no point for it being smaller
  // but it being larger may be beneficial with large ppc.
  // It should be a multiple of block dimension.
  for(auto chunk_idx = detail::bid(); chunk_idx < num_chunks;
      chunk_idx += detail::gdim()) {
    const auto chunk_offset = chunk_idx * chunk_size;
    deposit_current(
      num_box_candidates,
      cfl,
      charge,
      ids_span.subspan(chunk_offset, chunk_size),
      vel_span.subspan(chunk_offset, chunk_size),
      pos_span.subspan(chunk_offset, chunk_size),
      Jmds,
      lattice_origo_coordinates,
      scratch_span);
  }
}
}  // namespace pic
