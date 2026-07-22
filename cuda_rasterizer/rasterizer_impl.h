/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		// alignment 对齐, 比如是 128 字节, 对齐到最近的 128 字节的位置, 向上取
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1); 
		ptr = reinterpret_cast<T*>(offset); //后续用来取得内存存储的位置
		chunk = reinterpret_cast<char*>(ptr + count); // 指针运算, 因为指针 ptr 的类型是 T, 所以 chunk = ptr + count * sizeof(T)
	}

	// 用于存储渲染中高斯的一些几何性质
	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	// 模拟从 0 开始后续内存对齐的偏移量
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;  // 实际内存不是从 0 开始的, 要留有 128 字节的余量
	}
};