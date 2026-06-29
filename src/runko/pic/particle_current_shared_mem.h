// Copyright 2025 - 2026, Miro Palmu, Joonas Nättilä and the runko contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "runko/pic/particle.h"
#include "runko/tools/math.h"
#include "runko/tools/vector.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <hip/hip_runtime.h>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace deposit_kernel {
__host__ __device__ constexpr std::size_t
  num_warps(std::size_t num_threads, std::size_t warp_size)
{
  // Warp size is assumed to always be a power of two.
  // For Nvidia it's 32, for AMD 32 or 64.
  const bool even_multiple_of_warp_size = (num_threads & (warp_size - 1)) == 0;
  const auto warps_per_block            = num_threads / warp_size;
  return even_multiple_of_warp_size ? warps_per_block : warps_per_block + 1;
}

__device__ std::uint32_t
  warp_id()
{
  return threadIdx.x / warpSize;
}

__device__ std::uint32_t
  lane_id()
{
  return threadIdx.x & (warpSize - 1);
}

template<typename T, typename F>
__device__ T
  warp_reduce_to_lane_0(T t, F f)
{
  // Only lane 0 will contain the reduced value
  // N.B. Every lane must participate in this!
  // If some lanes are inactive, this'll produce incorrect results.
  for(auto src_lane = warpSize / 2u; src_lane >= 1u; src_lane /= 2u) {
    t = f(t, __shfl_down(t, src_lane));
  }

  return t;
}

template<typename T>
__device__ T
  prod(const toolbox::Vec3<T> &v)
{
  return v[0] * v[1] * v[2];
}

template<typename T, typename U>
__device__ T *
  align_up(U *ptr)
{
  // Either both are const or neither are const
  static_assert(not std::is_const_v<U> || std::is_const_v<T>);
  return reinterpret_cast<T *>(__builtin_align_up(ptr, std::alignment_of_v<T>));
}

static constexpr std::size_t vt_shared_mem_regions = 6ul;
static constexpr std::size_t bt_shared_mem_regions = 6ul;

template<typename value_type, typename bound_type>
__host__ __device__ constexpr std::size_t
  shared_mem_requirement(std::size_t num_threads, std::size_t num_warps)
{
  static constexpr auto vt_padding = std::alignment_of_v<value_type> - 1;
  static constexpr auto bt_padding = std::alignment_of_v<bound_type> - 1;
  const auto vt_mem = vt_shared_mem_regions * num_threads * sizeof(value_type);
  const auto bt_mem = bt_shared_mem_regions * num_warps * sizeof(bound_type);
  return vt_padding + vt_mem + bt_padding + bt_mem;
}

template<typename T, typename U>
__device__ std::span<T>
  reinterpret_span_or_empty(std::span<U> from)
{
  // If `from` is too small to fit either the padding, we'll return
  // an empty span
  auto *ptr          = deposit_kernel::align_up<T>(from.data());
  const auto padding = reinterpret_cast<std::uintptr_t>(ptr) -
                       reinterpret_cast<std::uintptr_t>(from.data());
  const auto total_bytes = from.size() * sizeof(U);
  if(padding >= total_bytes) { return std::span<T>(ptr, 0); }
  return std::span<T>(ptr, (total_bytes - padding) / sizeof(T));
}
}  // namespace deposit_kernel

namespace pic {
using bound_type = std::uint32_t;
template<typename T>
using Vec3 = toolbox::Vec3<T>;

template<typename value_type, typename JMDS, typename VMDS, typename IMDS>
__global__ void
  deposit_current_kernel(
    [[maybe_unused]] std::size_t num_shared_bytes,
    const value_type cfl,
    const value_type,
    IMDS ids_mds,
    VMDS vel_mds,
    VMDS pos_mds,
    JMDS,
    const std::array<value_type, 3> lattice_origo_coordinates)
{
  using Vec3v = Vec3<value_type>;
  using Vec3i = Vec3<std::uint32_t>;

  extern __shared__ std::byte scratch[];

  // value_type must be a floating point type
  static_assert(std::is_floating_point_v<value_type>);
  assert(
    (num_shared_bytes >= deposit_kernel::shared_mem_requirement<value_type, bound_type>(
                           blockDim.x,
                           deposit_kernel::num_warps(blockDim.x, warpSize))));

  // Each thread in the block shares these pointers.
  // They're used to store the x1 and x2 vectors that will be read from global memory
  // (pos & vel)
  const auto shared_xs_stride = blockDim.x;
  [[maybe_unused]] const std::array<value_type *, deposit_kernel::vt_shared_mem_regions>
    shared_xs = [&]() {
      auto *first = deposit_kernel::align_up<value_type>(scratch);
      std::array<value_type *, deposit_kernel::vt_shared_mem_regions> ptrs = {};
      for(auto &ptr: ptrs) {
        ptr = first;
        first += shared_xs_stride;
      }

      return ptrs;
    }();

  // This can coexist with shared_xs, these are stored after them.
  [[maybe_unused]] const std::array<bound_type *, deposit_kernel::bt_shared_mem_regions>
    shared_bounds = [&]() {
      // We're using the same shared memory arena for value_types and bound_types.
      // Here we align the memory after the last of value_type to be suitable for
      // bound_type.
      auto *first =
        deposit_kernel::align_up<bound_type>(shared_xs.back() + shared_xs_stride);
      std::array<bound_type *, deposit_kernel::bt_shared_mem_regions> ptrs = {};
      for(auto &ptr: ptrs) {
        ptr = first;
        first += deposit_kernel::num_warps(blockDim.x, warpSize);
      }

      return ptrs;
    }();

  // We're aliasing the same shared memory as the two arrays above.
  // This cannot coexist with them.
  [[maybe_unused]] auto box =
    deposit_kernel::reinterpret_span_or_empty<value_type, std::byte>(
      std::span<std::byte>(scratch, num_shared_bytes));

  const auto tid    = threadIdx.x + blockIdx.x * blockDim.x;
  const auto stride = blockDim.x * gridDim.x;
  // The warp reductions inside won't work if length of data is not
  // a multiple of warpSize, because the last threads of the last warp will not execute
  // the loop bode. Thus, we must loop over a length that is a multiple of the warpSize
  // and guard data access if the index is out of range.
  const auto misalignment = ids_mds.size() & (warpSize - 1ul);
  const auto end =
    misalignment > 0ul ? ids_mds.size() + warpSize - misalignment : ids_mds.size();

  for(auto idx = tid; idx < end; idx += stride) {
    // N.B. Must not use continue to skip dead particles before
    // we're done with warp reductions. Otherwise they can produce garbage

    auto guarded_access = [&](auto mds, auto otherwise) {
      return idx < mds.size() ? mds[idx][] : otherwise;
    };

    static constexpr value_type zero = value_type { 0 };
    static constexpr Vec3v zero_vec  = Vec3v { zero, zero, zero };

    const auto u = Vec3v(idx < vel_mds.size() ? vel_mds[idx] : zero_vec).template as<value_type>();
    const auto invgam =
      value_type { 1 } / sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));
    const auto x2 =
      Vec3v(idx < pos_mds.size() ? pos_mds[idx] : zero_vec).template as<value_type>() -
      Vec3v(lattice_origo_coordinates).template as<value_type>();
    const auto x1 = x2 - cfl * invgam * u;

    // Cache the computed/loaded x values in shared memory
    shared_xs[0][threadIdx.x] = x1[0];
    shared_xs[1][threadIdx.x] = x1[1];
    shared_xs[2][threadIdx.x] = x1[2];
    shared_xs[3][threadIdx.x] = x2[0];
    shared_xs[4][threadIdx.x] = x2[1];
    shared_xs[5][threadIdx.x] = x2[2];

    const auto i1 = x1.template as<bound_type>();
    const auto i2 = x2.template as<bound_type>();

    Vec3i aabb_min = { ~0u, ~0u, ~0u };
    Vec3i aabb_max = { 0u, 0u, 0u };

    if(guarded_access(ids_mds, runko::dead_prtc_id) != runko::dead_prtc_id) {
      // Only replace the reduction identities if we're dealing with a live particle
      for(auto i = 0u; i < 3u; i++) {
        aabb_min[i] = sstd::min(aabb_min[i], sstd::min(i1[i], i2[i]));
        aabb_max[i] = sstd::max(aabb_max[i], sstd::max(i1[i], i2[i]));
      }
    }

    // Each thread has their own bounding box. Let's bound all the boxes
    // so there's one block-wide bounding box, which we'll distribute
    // to each thread.

    // Gather the bounds from the lanes of the warp to lane 0
    for(auto i = 0u; i < 3u; i++) {
      aabb_min[i] =
        deposit_kernel::warp_reduce_to_lane_0(aabb_min[i], sstd::min<bound_type>);
      aabb_max[i] =
        deposit_kernel::warp_reduce_to_lane_0(aabb_max[i], sstd::max<bound_type>);
    }

    const auto wid       = deposit_kernel::warp_id();
    const auto num_warps = deposit_kernel::num_warps(blockDim.x, warpSize);

    // Lane 0 stores the bounds to scratch memory:
    if(0 == deposit_kernel::lane_id()) {
      for(auto i = 0u; i < 3u; i++) {
        shared_bounds[i][wid]      = aabb_min[i];
        shared_bounds[i + 3u][wid] = aabb_max[i];
      }
    }

    __syncthreads();

    // If num_warps is less than six (i.e. blockDim.x < 6 * 64), some warps
    // need to reduce more than one bound. If there are more than six warps,
    // the first six warps do the reductions in parallel.
    for(auto i = wid; i < 6u; i += num_warps) {
      auto *const bounds = shared_bounds[i];
      // First three warps reduce minimums, last three maximums
      bound_type val = bound_type { ~0u };
      auto f         = sstd::min<bound_type>;
      if(i >= 3u) {
        val = bound_type { 0 };
        f   = sstd::max<bound_type>;
      }

      // Only lanes with lane id < num_warps read the num_warps values from
      // scratch
      if(deposit_kernel::lane_id() < num_warps) {
        val = bounds[deposit_kernel::lane_id()];
      }

      // All lanes must participate in the reduction to produce correct values
      val = deposit_kernel::warp_reduce_to_lane_0(val, f);

      // Lane 0 stores the reduced bound back to scratch.
      // Store in the same location this warp read from to avoid data races
      // with other warps.
      if(0 == deposit_kernel::lane_id()) { bounds[0] = val; }
    }

    __syncthreads();

    // Finally, every thread of every warp reads the block-wide bounds
    // from scratch memory. After this, every thread has the box that bounds
    // all the points considered by the block.
    for(auto i = 0u; i < 3u; i++) {
      aabb_min[i] = shared_bounds[i][0];
      aabb_max[i] = shared_bounds[i + 3u][0];
    }

    // Need to keep in mind we might be working with idx values that are larger than the
    // data size!
  }
}
}  // namespace pic
//// This assumes cells can never be negative
// template<typename value_type>
// struct Box {
//   Vec3i min    = { ~0u, ~0u, ~0u };
//   Vec3i extent = { 0u, 0u, 0u };
//
//   // TODO:
//   // Inactive threads have ~0u in min and 0u in max
//   // extent will then be wrong
//   // Should also check max >= min, otherwise will underflow
//   // This should probably be rewritten:
//   // don't have box, just have aabb_min and aabb_max.
//   // Once we're done with reducing them, start using extent.
//   // Have them in the "higher" level function, and implement these
//   // box functions as free functions or lambdas.
//   __forceinline__ __device__ Box(const Vec3i& aabb_min, const Vec3i& aabb_max) :
//     min(aabb_min),
//     extent(aabb_max - min)
//   {
//   }
//
//   __forceinline__ __device__ bool contains(const Vec3i& point) const
//   {
//     bool contained = true;
// #pragma unroll
//     for(auto i = 0u; i < 3u; i++) {
//       contained &= min[i] <= point[i] && point[i] < min[i] + extent[i];
//     }
//
//     return contained;
//   }
//
//   __forceinline__ __device__ void bound_thread_boxes(Scratch scratch)
//   {
//   }
//
//   __forceinline__ __device__ std::size_t fit_to_scratch(Scratch scratch)
//   {
//     // The box is large enough to bound all the particles.
//     // It may be too large to fit into scratch memory, so it may have to
//     // be shrunk. We do that here.
//
//     // Increase size by two:
//     // If all the particles are in the same cell, the min and max bounds are
//     // the same. To make the upper bound strictly larger than any of the
//     // particle positions, increase it by one. The other one comes from the
//     // stencil operation to the J grid: we'll be adding current to the
//     // neighbouring cells in the positive directions, so we'll increase the
//     // size in each dimension by one.
//     extent          = extent + Vec3i { 2u, 2u, 2u };
//     auto volume     = deposit_kernel::prod(extent);
//     const auto size = scratch.size<bound_type>();
//
//     // We'll be storing three boxes in the same volume, because we'll be
//     // storing a current of Vec3 in the same scratch memory.
//     while(size < 3u * volume) {
//       const auto i = extent[0u] > extent[1u] ? (extent[0u] > extent[2u] ? 0u : 2u)
//                                              : (extent[1u] > extent[2u] ? 1u : 2u);
//       extent[i] -= 1u;
//       volume = deposit_kernel::prod(extent);
//     }
//
//     const auto count = 3u * volume;
//     return count;
//   }
//
//   __forceinline__ __device__ void
//     store(const Vec3i& point, const toolbox::Vec3<value_type>& current, Scratch
//     scratch)
//   {
//     const auto num_cells_per_component = scratch.size<value_type>() / 3u;
//     const auto delta                   = point - min;
//     const auto idx = delta[2] + delta[1] * extent[2] + delta[0] * extent[1] *
//     extent[2];
// #pragma unroll
//     for(auto i = 0u; i < 3u; i++) {
//       sstd::atomic_add(
//         &scratch.at<value_type>(idx + i * num_cells_per_component),
//         current[i]);
//     }
//   }
//
//   template<typename JMDS>
//   __forceinline__ __device__ void copy_from_shared_to_global(JMDS Jmds, Scratch
//   scratch)
//   {
//     const auto num_cells_per_component = scratch.size<value_type>() / 3u;
//     for(auto shmem_idx = deposit_kernel::tid(); shmem_idx < num_cells_per_component;
//         shmem_idx += deposit_kernel::bdim()) {
//       const auto si = (Vec3i { shmem_idx / (extent[1] * extent[2]),
//                                (shmem_idx / extent[2]) % extent[1],
//                                shmem_idx % extent[2] } +
//                        min)
//                         .as<runko::index_t>();
//       value_type* const J[3] = {
//         &thrust::raw_reference_cast(Jmds[si.data][0]),
//         &thrust::raw_reference_cast(Jmds[si.data][1]),
//         &thrust::raw_reference_cast(Jmds[si.data][2]),
//       };
//
// #pragma unroll
//       for(auto i = 0u; i < 3u; i++) {
//         const auto val =
//           scratch.at<value_type>(shmem_idx + i * num_cells_per_component);
//         if(0 != val) { sstd::atomic_add(J[i], val); }
//       }
//     }
//
//     __syncthreads();
//   }
// };
//
// template<typename value_type, typename JMDS, typename VMDS, typename IMDS>
//__device__ void
//   deposit_current(
//     std::uint32_t num_box_candidates,
//     std::uint32_t chunk_offset,
//     std::uint32_t chunk_size,
//     const value_type cfl,
//     const value_type charge,
//     IMDS ids_mds,
//     VMDS vel_mds,
//     VMDS pos_mds,
//     JMDS Jmds,
//     const std::array<value_type, 3> lattice_origo_coordinates,
//     std::span<std::byte> scratch_memory)
//{
//   using Vec3v = toolbox::Vec3<value_type>;
//
//   // If num_box_candidates is smaller than the chunk size, we'll construct
//   // the bounding box considering only a part of the particles in the chunk.
//   const auto count                = sstd::min(chunk_size, num_box_candidates);
//   const auto [aabb_min, aabb_max] = compute_thread_bounds(
//     cfl,
//     chunk_offset,
//     count,
//     ids_mds,
//     vel_mds,
//     pos_mds,
//     lattice_origo_coordinates);
//
//   Scratch scratch = { scratch_memory };
//   [aabb_min]      = bound_thread_boxes(scratch);
//
//   Box<value_type> box(aabb_min, aabb_max);
//   const auto new_size = box.fit_to_scratch(scratch);
//   scratch.resize<value_type>(new_size);
//   scratch.set_to_zero();
//
//   const auto store_current = [&](const Vec3i& point, const Vec3v& current) {
//     if(box.contains(point)) {
//       box.store(point, current, scratch);
//     } else {
//       const auto si  = point.template as<runko::index_t>();
//       auto* const Jx = &thrust::raw_reference_cast(Jmds[si.data][0]);
//       auto* const Jy = &thrust::raw_reference_cast(Jmds[si.data][1]);
//       auto* const Jz = &thrust::raw_reference_cast(Jmds[si.data][2]);
//
//       sstd::atomic_add(Jx, current[0]);
//       sstd::atomic_add(Jy, current[1]);
//       sstd::atomic_add(Jz, current[2]);
//     }
//   };
//
//   for(auto elem_idx = chunk_offset + deposit_kernel::tid();
//       elem_idx < chunk_offset + chunk_size;
//       elem_idx += deposit_kernel::bdim()) {
//     if(ids_mds[elem_idx][] == runko::dead_prtc_id) { continue; }
//
//     const auto u = Vec3v(vel_mds[elem_idx]).template as<value_type>();
//     const auto invgam =
//       value_type { 1 } / sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));
//
//     const auto x2 = Vec3v(pos_mds[elem_idx]).template as<value_type>() -
//                     Vec3v(lattice_origo_coordinates).template as<value_type>();
//
//     const auto x1 = x2 - cfl * invgam * u;
//
//     // Float floor for relay and weight computation (pure float — no 64-bit
//     // integers)
//     const auto fi1 = Vec3v(sstd::floor(x1(0)), sstd::floor(x1(1)),
//     sstd::floor(x1(2))); const auto fi2 = Vec3v(sstd::floor(x2(0)),
//     sstd::floor(x2(1)), sstd::floor(x2(2)));
//
//     const auto relay = [&](const runko::index_t j) -> value_type {
//       const auto a  = sstd::min(fi1(j), fi2(j)) + value_type { 1 };
//       const auto b1 = sstd::max(fi1(j), fi2(j));
//       const auto b2 = value_type { 0.5 } * (x1(j) + x2(j));
//       const auto b  = sstd::max(b1, b2);
//       return sstd::min(a, b);
//     };
//
//     const auto x_relay = Vec3v(relay(0), relay(1), relay(2));
//
//     const auto F1 = charge * (x_relay - x1);
//     const auto F2 = charge * (x2 - x_relay);
//
//     const auto i1 = fi1.template as<uint32_t>();
//     const auto i2 = fi2.template as<uint32_t>();
//
//     const auto W1 = value_type { 0.5 } * (x1 + x_relay) - fi1;
//     const auto W2 = value_type { 0.5 } * (x2 + x_relay) - fi2;
//
//     const auto [Fx1, Fy1, Fz1] = F1.data;
//     const auto [Fx2, Fy2, Fz2] = F2.data;
//     const auto [Wx1, Wy1, Wz1] = W1.data;
//     const auto [Wx2, Wy2, Wz2] = W2.data;
//
//     static constexpr auto one = value_type { 1 };
//
//     store_current(
//       i1,
//       Vec3v(
//         Fx1 * (one - Wy1) * (one - Wz1),
//         Fy1 * (one - Wx1) * (one - Wz1),
//         Fz1 * (one - Wx1) * (one - Wy1)));
//
//     store_current(
//       i2,
//       Vec3v(
//         Fx2 * (one - Wy2) * (one - Wz2),
//         Fy2 * (one - Wx2) * (one - Wz2),
//         Fz2 * (one - Wx2) * (one - Wy2)));
//     store_current(
//       i1 + Vec3i(1, 0, 0),
//       Vec3v(0, Fy1 * Wx1 * (one - Wz1), Fz1 * Wx1 * (one - Wy1)));
//     store_current(
//       i2 + Vec3i(1, 0, 0),
//       Vec3v(0, Fy2 * Wx2 * (one - Wz2), Fz2 * Wx2 * (one - Wy2)));
//     store_current(
//       i1 + Vec3i(0, 1, 0),
//       Vec3v(Fx1 * Wy1 * (one - Wz1), 0, Fz1 * (one - Wx1) * Wy1));
//     store_current(
//       i2 + Vec3i(0, 1, 0),
//       Vec3v(Fx2 * Wy2 * (one - Wz2), 0, Fz2 * (one - Wx2) * Wy2));
//     store_current(
//       i1 + Vec3i(0, 0, 1),
//       Vec3v(Fx1 * (one - Wy1) * Wz1, Fy1 * (one - Wx1) * Wz1, 0));
//     store_current(
//       i2 + Vec3i(0, 0, 1),
//       Vec3v(Fx2 * (one - Wy2) * Wz2, Fy2 * (one - Wx2) * Wz2, 0));
//     store_current(i1 + Vec3i(0, 1, 1), Vec3v(Fx1 * Wy1 * Wz1, 0, 0));
//     store_current(i2 + Vec3i(0, 1, 1), Vec3v(Fx2 * Wy2 * Wz2, 0, 0));
//     store_current(i1 + Vec3i(1, 0, 1), Vec3v(0, Fy1 * Wx1 * Wz1, 0));
//     store_current(i2 + Vec3i(1, 0, 1), Vec3v(0, Fy2 * Wx2 * Wz2, 0));
//     store_current(i1 + Vec3i(1, 1, 0), Vec3v(0, 0, Fz1 * Wx1 * Wy1));
//     store_current(i2 + Vec3i(1, 1, 0), Vec3v(0, 0, Fz2 * Wx2 * Wy2));
//   }
//
//   __syncthreads();
//
//   box.copy_from_shared_to_global(Jmds, scratch);
// }
//
// template<typename value_type, typename JMDS, typename VMDS, typename IMDS>
//__global__ void
//   deposit_current_kernel_old(
//     std::uint32_t num_chunks,
//     std::uint32_t chunk_size,
//     std::uint32_t num_shared_bytes,
//     std::uint32_t num_box_candidates,
//     const value_type cfl,
//     const value_type charge,
//     IMDS ids_mds,
//     VMDS vel_mds,
//     VMDS pos_mds,
//     JMDS Jmds,
//     const std::array<value_type, 3> lattice_origo_coordinates)
//{
//   static_assert(std::is_floating_point_v<value_type>);
//   extern __shared__ std::byte scratch[];
//
//   // At least a 2x2x2 box of Vec3 must fit into scratch
//   // box creation also uses scratch and requires 6 * num_warps *
//   // sizeof(bound_type) bytes
//   assert(num_shared_bytes >= 3u * 2u * 2u * 2u * sizeof(value_type));
//   assert(num_shared_bytes >= 6u * deposit_kernel::num_warps() * sizeof(bound_type));
//
//   std::span<std::byte> scratch_memory(scratch, num_shared_bytes);
//
//   // Work over chunks: each block goes over a chunk and chunk size may
//   // be different from block dimension. There's no point for it being smaller
//   // but it being larger may be beneficial with large ppc.
//   // It should be a multiple of block dimension.
//   for(auto chunk_idx = deposit_kernel::bid(); chunk_idx < num_chunks;
//       chunk_idx += deposit_kernel::gdim()) {
//     const auto chunk_offset = chunk_idx * chunk_size;
//     deposit_current(
//       num_box_candidates,
//       chunk_offset,
//       chunk_size,
//       cfl,
//       charge,
//       ids_mds,
//       vel_mds,
//       pos_mds,
//       Jmds,
//       lattice_origo_coordinates,
//       scratch_memory);
//   }
// }
//
