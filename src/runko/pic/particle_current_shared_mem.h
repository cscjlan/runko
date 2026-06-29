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

namespace detail {
__forceinline__ __device__ decltype(auto)
  num_warps()
{
  // Warp size is assumed to always be a power of two.
  // For Nvidia it's 32, for AMD 32 or 64.
  const bool even_multiple_of_warp_size = (blockDim.x & (warpSize - 1)) == 0;
  const auto warps_per_block            = blockDim.x / warpSize;
  return even_multiple_of_warp_size ? warps_per_block : warps_per_block + 1;
}

__forceinline__ __device__ decltype(auto)
  warp_id()
{
  return threadIdx.x / warpSize;
}

__forceinline__ __device__ decltype(auto)
  lane_id()
{
  return threadIdx.x & (warpSize - 1);
}

template<typename T, typename F>
__forceinline__ __device__ T
  warp_reduce_to_lane_0(T t, F f)
{
  // Only lane 0 will contain the reduced value
#pragma unroll
  for(auto src_lane = warpSize; src_lane >= 1; src_lane /= 2) {
    t = f(t, __shfl_down(t, src_lane));
  }

  return t;
}

template<typename T>
__forceinline__ __device__ T
  prod(const toolbox::Vec3<T> &v)
{
  return v[0] * v[1] * v[2];
}
}  // namespace detail

namespace pic {
using bound_type = std::uint32_t;
template<typename T>
using Vec3 = toolbox::Vec3<T>;

// struct Scratch {
// private:
//   std::span<std::byte> scratch;
//
//   template<typename T>
//   __forceinline__ __device__ decltype(auto) padding_to_alignment() const
//   {
//     static constexpr auto alignment = std::alignment_of_v<T>;
//     const auto address              =
//     reinterpret_cast<std::uintptr_t>(scratch.data()); const auto bytes_over_alignment
//     = address & (alignment - 1ul); const auto padding =
//       bytes_over_alignment > 0ul ? alignment - bytes_over_alignment : 0ul;
//     return padding;
//   }
//
// public:
//   template<typename T>
//   __forceinline__ __device__ T* data()
//   {
//     return static_cast<T*>(
//       static_cast<void*>(scratch.data() + padding_to_alignment<T>()));
//   }
//
//   template<typename T>
//   __forceinline__ __device__ const T* data() const
//   {
//     return static_cast<const T*>(
//       static_cast<const void*>(scratch.data() + padding_to_alignment<T>()));
//   }
//
//   template<typename T>
//   __forceinline__ __device__ T& at(std::size_t i)
//   {
//     return data<T>()[i];
//   }
//
//   template<typename T>
//   __forceinline__ __device__ const T& at(std::size_t i) const
//   {
//     return data<T>()[i];
//   }
//
//   template<typename T>
//   __forceinline__ __device__ std::size_t size() const
//   {
//     return (scratch.size() - padding_to_alignment<T>()) / sizeof(T);
//   }
//
//   template<typename T>
//   __forceinline__ __device__ void resize(std::size_t i)
//   {
//     std::byte* end = static_cast<std::byte*>(static_cast<void*>(data<T>() + i));
//     scratch        = std::span<std::byte>(
//       scratch.data(),
//       static_cast<std::size_t>(end - scratch.data()));
//   }
//
//   __forceinline__ __device__ void set_to_zero()
//   {
//     for(auto i = detail::tid(); i < scratch.size(); i += detail::bdim()) {
//       scratch.data()[i] = std::byte { 0 };
//     }
//     __syncthreads();
//   }
// };
//
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
//     // Each thread has their own bounding box. This function
//     // bounds all the individual bounding boxes to a block-wide bounding box,
//     // which each of the threads will then share.
//
//     // Gather the bounds from the lanes of the warp to lane 0
//     Vec3i aabb_max = min + extent;
// #pragma unroll
//     for(auto i = 0u; i < 3u; i++) {
//       min[i] = detail::warp_reduce_to_lane_0(min[i], sstd::min<bound_type>);
//       aabb_max[i] =
//         detail::warp_reduce_to_lane_0(aabb_max[i], sstd::max<bound_type>);
//     }
//
//     // Lane 0 stores the bounds to scratch memory:
//     // First num_warps values contain min[0] for each warp,
//     // next num_warps values contain min[1] for each warp,
//     // then min[2],
//     // then max[0] and so on.
//     if(0 == detail::lid()) {
// #pragma unroll
//       for(auto i = 0u; i < 3u; i++) {
//         scratch.at<bound_type>(detail::wid() + i * detail::num_warps()) = min[i];
//         scratch.at<bound_type>(detail::wid() + (3u + i) * detail::num_warps()) =
//           aabb_max[i];
//       }
//     }
//
//     __syncthreads();
//
//     // If num_warps is less than six (i.e. blockDim.x < 6 * 64), some warps
//     // need to reduce more than one bound. If there are more than six warps,
//     // the first six warps do the reductions is parallel.
//     for(auto i = detail::wid(); i < 6u; i += detail::num_warps()) {
//       // Each thread participates to avoid deadlock with syncthreads
//       bound_type val = bound_type {0};
//
//       // Only lanes with lane id < num_warps read the num_warps values from
//       // scratch
//       if(detail::lid() < detail::num_warps()) {
//         val = scratch.at<bound_type>(detail::lid() + i * detail::num_warps());
//       }
//       const auto f = i < 3u ? sstd::min<bound_type> : sstd::max<bound_type>;
//
//       // All lanes participate, but lane 0 only contains reductions from first
//       // num_warps lanes. This assumes num_warps is a power of two.
//       // N.B. block size must be warpSize * 2^k
//       for(auto j = detail::num_warps() / 2u; j >= 1u; j /= 2u) {
//         val = f(val, __shfl_down(val, j));
//       }
//
//       // Lane 0 stores the reduced bound back to scratch.
//       // Store in the same location this warp read from to avoid data races
//       // with other warps.
//       if(0 == detail::lid()) {
//         scratch.at<bound_type>(i * detail::num_warps()) = val;
//       }
//     }
//
//     __syncthreads();
//
//     // Finally, every thread of every warp reads the block-global bounds
//     // from scratch memory. After this, every thread has the box that bounds
//     // all the points considered by the block.
// #pragma unroll
//     for(auto i = 0u; i < 3u; i++) {
//       min[i]      = scratch.at<bound_type>(i * detail::num_warps());
//       aabb_max[i] = scratch.at<bound_type>((i + 3u) * detail::num_warps());
//     }
//
//     extent = aabb_max - min;
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
//     auto volume     = detail::prod(extent);
//     const auto size = scratch.size<bound_type>();
//
//     // We'll be storing three boxes in the same volume, because we'll be
//     // storing a current of Vec3 in the same scratch memory.
//     while(size < 3u * volume) {
//       const auto i = extent[0u] > extent[1u] ? (extent[0u] > extent[2u] ? 0u : 2u)
//                                              : (extent[1u] > extent[2u] ? 1u : 2u);
//       extent[i] -= 1u;
//       volume = detail::prod(extent);
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
//     for(auto shmem_idx = detail::tid(); shmem_idx < num_cells_per_component;
//         shmem_idx += detail::bdim()) {
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
// template<typename value_type, typename VMDS, typename IMDS>
//__forceinline__ __device__ std::pair<Vec3i, Vec3i>
//   compute_thread_bounds(
//     const value_type cfl,
//     std::uint32_t offset,
//     std::uint32_t count,
//     IMDS ids_mds,
//     VMDS vel_mds,
//     VMDS pos_mds,
//     const std::array<value_type, 3> lattice_origo_coordinates)
//{
//   using Vec3v = toolbox::Vec3<value_type>;
//
//   Vec3i aabb_min = { ~0u, ~0u, ~0u };
//   Vec3i aabb_max = { 0u, 0u, 0u };
//   for(auto elem_idx = offset + detail::tid(); elem_idx < offset + count;
//       elem_idx += detail::bdim()) {
//     if(ids_mds[elem_idx][] == runko::dead_prtc_id) { continue; }
//
//     const auto u = Vec3v(vel_mds[elem_idx]).template as<value_type>();
//     const auto invgam =
//       value_type { 1 } / sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));
//     const auto x2 = Vec3v(pos_mds[elem_idx]).template as<value_type>() -
//                     Vec3v(lattice_origo_coordinates).template as<value_type>();
//     const auto x1 = x2 - cfl * invgam * u;
//     const auto i1 = x1.template as<bound_type>();
//     const auto i2 = x2.template as<bound_type>();
//
// #pragma unroll
//     for(auto i = 0u; i < 3; i++) {
//       aabb_min[i] = sstd::min(aabb_min[i], sstd::min(i1[i], i2[i]));
//       aabb_max[i] = sstd::max(aabb_max[i], sstd::max(i1[i], i2[i]));
//     }
//   }
//
//   return std::make_pair(aabb_min, aabb_max);
// }
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
//   for(auto elem_idx = chunk_offset + detail::tid();
//       elem_idx < chunk_offset + chunk_size;
//       elem_idx += detail::bdim()) {
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
//   assert(num_shared_bytes >= 6u * detail::num_warps() * sizeof(bound_type));
//
//   std::span<std::byte> scratch_memory(scratch, num_shared_bytes);
//
//   // Work over chunks: each block goes over a chunk and chunk size may
//   // be different from block dimension. There's no point for it being smaller
//   // but it being larger may be beneficial with large ppc.
//   // It should be a multiple of block dimension.
//   for(auto chunk_idx = detail::bid(); chunk_idx < num_chunks;
//       chunk_idx += detail::gdim()) {
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

template<typename value_type, typename JMDS, typename VMDS, typename IMDS>
__global__ void
  deposit_current_kernel(
    [[maybe_unused]] std::uint32_t num_shared_bytes,
    const value_type,
    const value_type,
    IMDS ids_mds,
    VMDS,
    VMDS,
    JMDS,
    const std::array<value_type, 3>)
{
  extern __shared__ std::byte scratch[];

  [&]() {
    // value_type must be a floating point type
    static_assert(std::is_floating_point_v<value_type>);
    // value_type size must be larger or equal to bound_type size
    // makes alignment in shared memory easier
    static_assert(sizeof(value_type) >= sizeof(bound_type));

    // We need to be able to store the x1 and x2 vec3 values
    // for each thread,
    // as well as the 6 bounds (3 for min, 3 for max)
    // for all warps.
    [[maybe_unused]] const std::uint32_t shared_mem_requirement =
      6u * (detail::num_warps() * sizeof(bound_type) + blockDim.x * sizeof(value_type));
    assert(num_shared_bytes >= shared_mem_requirement);
  }();

  [[maybe_unused]] auto align_up = []<typename T, typename U>(U *ptr) {
    return reinterpret_cast<T *>(__builtin_align_up(ptr, std::alignment_of_v<T>));
  };

  [[maybe_unused]] const std::array<value_type *, 6> shared_x_ptrs = [&]() {
    std::array<value_type *, 6> ptrs = {
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    };
    ptrs[0] = align_up.template operator()<value_type>(scratch);
#pragma unroll
    for(auto i = 1ul; i < 6ul; i++) {
      ptrs[i] = align_up.template operator()<value_type>(ptrs[i - 1ul]);
    }

    return ptrs;
  }();

  const auto tid    = threadIdx.x + blockIdx.x * blockDim.x;
  const auto stride = blockDim.x * gridDim.x;
  for(auto idx = tid; idx < ids_mds.size(); idx += stride) {}
}
}  // namespace pic
