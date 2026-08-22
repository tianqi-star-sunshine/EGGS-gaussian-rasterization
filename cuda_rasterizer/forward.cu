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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir); 

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];  // 0 阶球谐函数

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;  // 在 python 端初始化时减去了 0.5, 在渲染颜色的时候需要将这 0.5 加回来

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	// 记录哪些通道的颜色为负
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);

	// 将负通道的颜色截断, clamped 相当于记录数组, 在反向传播的过程中
    // 如果颜色被截断过, 让被截断的梯度为 0  
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// 计算雅可比矩阵
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 把世界坐标系变成相机坐标系
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// 计算 2D 协方差矩阵
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
// 计算协方差矩阵
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Compute the projective mapping from the local 2D Gaussian plane
// (u, v, 1) to homogeneous pixel coordinates.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec3 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H,
	glm::mat3& T,
	float3& normal)
{
	const float inv_norm = rsqrtf(max(
		rot.x * rot.x + rot.y * rot.y + rot.z * rot.z + rot.w * rot.w,
		1e-12f));
	const float r = rot.x * inv_norm;
	const float x = rot.y * inv_norm;
	const float y = rot.z * inv_norm;
	const float z = rot.w * inv_norm;

	// GLM constructors are column-major. The first two columns span the
	// surfel plane; the third column is its unit normal.
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y + r * z),
		2.f * (x * z - r * y),
		2.f * (x * y - r * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z + r * x),
		2.f * (x * z + r * y),
		2.f * (y * z - r * x),
		1.f - 2.f * (x * x + y * y));

	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	const glm::mat3 L = R * S;

	const glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0f),
		glm::vec4(L[1], 0.0f),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1.0f));

	const glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]);

	const glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0f, 0.0f, 0.0f, float(W - 1) / 2.0f),
		glm::vec4(0.0f, float(H) / 2.0f, 0.0f, float(H - 1) / 2.0f),
		glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3(
		make_float3(L[2].x, L[2].y, L[2].z), viewmatrix);
}

/**
 * @brief: 计算 2D 高斯的包围盒
 */
__device__ bool compute_aabb(
	glm::mat3 T,
	float cutoff,
	float2& point_image, // 投影椭圆 AABB 的中心
	float2& extent //
) {
	// 对偶二次型
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	// d 对应的 A
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		// B_x / A , 对应的是 p_x , 也就是横向中心
		glm::dot(f, T[0] * T[2]),
		// 对应的纵向中心
		glm::dot(f, T[1] * T[2])
	);

	// 分别计算横向和纵向半宽的平方
	glm::vec2 h0 = p * p -
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0)); // 计算半宽

	// 得到轴对齐包围盒
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

/**
 * @brief: 2D 高斯的预处理
 * @brief: 1、计算 transMat 2、计算包围盒 3、计算颜色 4、法向量对齐 5、计算高斯覆盖多少 tile
 */
template<int C>
__device__ void preprocess_2d(
    int idx,
    const float3& p_view,
    // --- 几何与变换输入 ---
    const float* orig_points,
    const glm::vec3* scales,
    float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* transMat_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    int W, int H,
    // --- 颜色输入 ---
	// D 对应的阶数, M 对应的最大系数数量
    int D, int M,
    const glm::vec3* cam_pos,
    const float* shs,
    bool* clamped,
    const float* colors_precomp,
    // --- tile 网格 ---
    const dim3 grid,
    // --- 输出 ---
    float* transMats,
    float* rgb,
    float* depths,
    int* radii,
    float2* points_xy_image,
    float4* normal_opacity,
    uint32_t* tiles_touched)
{
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		// 计算从局部坐标系变换到屏幕坐标系
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0],
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		glm::mat3 unused_T;
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier,
			rotations[idx], projmatrix, viewmatrix, W, H, unused_T, normal);
	}

	// Always populate the internal transform buffer. The render stage reads
	// this buffer regardless of whether T was computed or supplied.
	float3* T_ptr = (float3*)transMats;
	T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
	T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
	T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};

// 
#if DUAL_VISIABLE
	float cos = -(p_view.x * normal.x + p_view.y * normal.y + p_view.z * normal.z);
	if (fabsf(cos) < 1e-8f) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = make_float3(
		multiplier * normal.x,
		multiplier * normal.y,
		multiplier * normal.z);
#endif

#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float cutoff = sqrtf(max(9.f + 2.f * logf(opacities[idx]), 0.000001));
#else
	float cutoff = 3.0f; // 截断值为 3 个标准差, 这是在圆形高斯空间中进行
#endif

	// Compute center and radius
	// 计算包围盒的 radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, cutoff, point_image, extent);
		if (!ok) return;
		radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
	}

	// 获得覆盖的 tile 的范围
	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// 计算当前高斯的 rgb 的颜色
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;  // 深度
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

/**
 * @brief: 3D 高斯的预处理
 * @brief: 1、计算 3D/2D 协方差 2、计算包围盒 3、计算颜色 4、计算高斯覆盖多少 tile
 */
template<int C>
__device__ void preprocess_3d(
	int idx,
	const float3& p_view,
	// --- 几何与变换输入 ---
	const float* orig_points,
	const glm::vec3* scales,
	float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	int W, int H,
	float tan_fovx, float tan_fovy,
	float focal_x, float focal_y,
	// --- 颜色输入 ---
	int D, int M,
	const glm::vec3* cam_pos,
	const float* shs,
	bool* clamped,
	const float* colors_precomp,
	// --- tile 网格 ---
	const dim3 grid,
	// --- 输出 ---
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	uint32_t* tiles_touched,
	bool antialiasing)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix.
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if (antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov));

	const float det = det_cov_plus_h_cov;
	if (det == 0.0f)
		return;

	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));

	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	conic_opacity[idx] = {
		conic.x,
		conic.y,
		conic.z,
		opacities[idx] * h_convolution_scaling
	};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);  // 计算单个高斯覆盖的 tile 数量
}

/**
 * @brief: 开发 hybrid CUDA kernel
 */
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations, // rotations 中维护的是一个四元数
	const float* opacities,
	const uint8_t* gaussian_type,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,  // 投影矩阵
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	float* transMats,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	// calculate the global index
	auto idx = cg::this_grid().thread_rank();  // 这里的 idx 是线程在 grid 中的全局编号
	if (idx >= P) // P 代表高斯的数量
		return;

	// Invisible Gaussians must leave deterministic zero-sized footprints.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	float3 p_view;
	// 视锥裁剪
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	const bool is_2d = gaussian_type[idx] == GAUSSIAN_2D;
	if (is_2d) {
		preprocess_2d<C>(
			idx,
			p_view,
			orig_points,
			scales,
			scale_modifier,
			rotations,
			opacities,
			transMat_precomp,
			viewmatrix,
			projmatrix,
			W, H,
			D, M,
			cam_pos,
			shs,
			clamped,
			colors_precomp,
			grid,
			transMats,
			rgb,
			depths,
			radii,
			points_xy_image,
			normal_opacity,
				tiles_touched
			);
	} else {
		preprocess_3d<C>(
			idx,
			p_view,
			orig_points,
			scales,
			scale_modifier,
			rotations,
			opacities,
			cov3D_precomp,
			viewmatrix,
			projmatrix,
			W, H,
			tan_fovx, tan_fovy,
			focal_x, focal_y,
			D, M,
			cam_pos,
			shs,
			clamped,
			colors_precomp,
			grid,
			radii,
			points_xy_image,
			depths,
			cov3Ds,
			rgb,
			conic_opacity,
			tiles_touched,
			antialiasing
		);
	}
}



// // Perform initial steps for each Gaussian prior to rasterization.
// /**
//  * @brief: 计算前向渲染部分之前的预处理
//  * @brief: 对高斯进行预处理, 离相机太近的高斯滤掉
//  * @brief: 计算每个高斯的 3D 协方差、2D 协方差
//  * @brief: 通过 SH 系数以及阶数计算每个高斯的颜色
//  * @brief: 计算协方差的逆, 计算每个高斯的包围盒
//  */
// template<int C>
// __global__ void preprocessCUDA(int P, int D, int M,
// 	const float* orig_points,
// 	const glm::vec3* scales,
// 	const float scale_modifier,
// 	const glm::vec4* rotations, // rotations 中维护的是一个四元数
// 	const float* opacities,
// 	const uint8_t* gaussian_type,
// 	const float* shs,
// 	bool* clamped,
// 	const float* cov3D_precomp,
// 	const float* colors_precomp,
// 	const float* viewmatrix,
// 	const float* projmatrix,
// 	const glm::vec3* cam_pos,
// 	const int W, int H,
// 	const float tan_fovx, float tan_fovy,
// 	const float focal_x, float focal_y,
// 	int* radii,
// 	float2* points_xy_image,
// 	float* depths,
// 	float* cov3Ds,
// 	float* rgb,
// 	float4* conic_opacity,
// 	const dim3 grid,
// 	uint32_t* tiles_touched,
// 	bool prefiltered,
// 	bool antialiasing)
// {
// 	auto idx = cg::this_grid().thread_rank();  // 这里的 idx 是线程在 grid 中的全局编号
// 	if (idx >= P) // P 代表高斯的数量
// 		return;

// 	// Initialize radius and touched tiles to 0. If this isn't changed,
// 	// this Gaussian will not be processed further.
// 	radii[idx] = 0;
// 	tiles_touched[idx] = 0;

// 	// Perform near culling, quit if outside.
// 	// 对距离太近的点进行删除
// 	float3 p_view;
// 	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
// 		return;

// 	// Transform point by projecting
// 	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
// 	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
// 	float p_w = 1.0f / (p_hom.w + 0.0000001f);
// 	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w }; // 高斯中心点投影后的齐次坐标

// 	// If 3D covariance matrix is precomputed, use it, otherwise compute
// 	// from scaling and rotation parameters.
// 	const float* cov3D;
// 	if (cov3D_precomp != nullptr)
// 	{
// 		cov3D = cov3D_precomp + idx * 6;
// 	}
// 	else
// 	{
// 		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
// 		cov3D = cov3Ds + idx * 6;
// 	}

// 	// Compute 2D screen-space covariance matrix
// 	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

// 	// 由于给 2D 高斯的对角线 + 0.3 , 整体的 det 变大了
// 	// 低通滤波
// 	constexpr float h_var = 0.3f;
// 	const float det_cov = cov.x * cov.z - cov.y * cov.y;
// 	cov.x += h_var;
// 	cov.z += h_var;
// 	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
// 	float h_convolution_scaling = 1.0f;

// 	// 因为整体的 det 增大
// 	if(antialiasing)
// 		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

// 	// Invert covariance (EWA algorithm)
// 	const float det = det_cov_plus_h_cov;

// 	// calculate the conic matrix
// 	if (det == 0.0f)
// 		return;
// 	float det_inv = 1.f / det;
// 	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

// 	// Compute extent in screen space (by finding eigenvalues of
// 	// 2D covariance matrix). Use extent to compute a bounding rectangle
// 	// of screen-space tiles that this Gaussian overlaps with. Quit if
// 	// rectangle covers 0 tiles.

// 	// 计算特征值
// 	float mid = 0.5f * (cov.x + cov.z);
// 	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
// 	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));

// 	// 高斯椭圆的特征值代表方差
// 	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2))); // calculate sigma -> 标准差, 3 sigma
// 	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
// 	uint2 rect_min, rect_max;
// 	getRect(point_image, my_radius, rect_min, rect_max, grid);
// 	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
// 		return;

// 	// If colors have been precomputed, use them, otherwise convert
// 	// spherical harmonics coefficients to RGB color.
// 	if (colors_precomp == nullptr)
// 	{
// 		// 从球谐函数以及系数计算颜色, 维护的 rgb 一维数组
// 		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
// 		rgb[idx * C + 0] = result.x;
// 		rgb[idx * C + 1] = result.y;
// 		rgb[idx * C + 2] = result.z;
// 	}

// 	// Store some useful helper data for the next steps.
// 	depths[idx] = p_view.z;
// 	radii[idx] = my_radius; // 计算出来的每个高斯的 3 sigma, 用来计算高斯的方形包围盒
// 	points_xy_image[idx] = point_image;
// 	// Inverse 2D covariance and opacity neatly pack into one float4
// 	float opacity = opacities[idx];

// 	// conic_opacity 中存储的是 协方差的逆以及抗混叠之后的不透明度
// 	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };

// 	// 覆盖的 tiles 的数量
// 	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
// }

/**
 * @brief: 单个 Gaussian 对当前像素的渲染响应。
 * @brief: 两种渲染模式只负责计算响应，alpha blending 由 renderCUDA 统一完成。
 */
struct RenderEval
{
	float power;
	float opacity;
	float depth;
	bool valid;
};

/**
 * @brief: 3D Gaussian 的像素响应。
 * @brief: 使用屏幕空间二维逆协方差计算马氏距离。
 */
__forceinline__ __device__ RenderEval render_mode_3d(
	const float2& pixf,
	const float2& point_image,
	const float4& conic_opacity,
	float center_depth)
{
	RenderEval eval;

	const float2 d = {
		point_image.x - pixf.x,
		point_image.y - pixf.y
	};

	// conic_opacity.xyz stores the symmetric inverse covariance
	// [A B; B C], so d^T Sigma^-1 d = A dx^2 + 2B dxdy + C dy^2.
	const float rho =
		conic_opacity.x * d.x * d.x
		+ 2.0f * conic_opacity.y * d.x * d.y
		+ conic_opacity.z * d.y * d.y;

	eval.power = -0.5f * rho;
	eval.opacity = conic_opacity.w;
	eval.depth = center_depth;
	eval.valid = eval.power <= 0.0f;
	return eval;
}

/**
 * @brief: 2D Gaussian 的像素响应。
 * @brief: 先求像素射线与 Gaussian 局部平面的交点，再在局部 (u, v) 平面计算距离。
 */
__forceinline__ __device__ RenderEval render_mode_2d(
	const float2& pixf,
	const float2& point_image,
	const float3& Tu,
	const float3& Tv,
	const float3& Tw,
	const float4& normal_opacity)
{
	RenderEval eval;
	eval.power = 0.0f;
	eval.opacity = normal_opacity.w;
	eval.depth = 0.0f;
	eval.valid = false;

	// The pixel imposes two homogeneous plane constraints in the local
	// Gaussian coordinates. Their cross product gives the intersection.
	const float3 k = {
		pixf.x * Tw.x - Tu.x,
		pixf.x * Tw.y - Tu.y,
		pixf.x * Tw.z - Tu.z
	};
	const float3 l = {
		pixf.y * Tw.x - Tv.x,
		pixf.y * Tw.y - Tv.y,
		pixf.y * Tw.z - Tv.z
	};
	const float3 p = {
		k.y * l.z - k.z * l.y,
		k.z * l.x - k.x * l.z,
		k.x * l.y - k.y * l.x
	};

	if (fabsf(p.z) < 1e-8f)
		return eval;

	const float2 uv = {p.x / p.z, p.y / p.z};
	const float rho_surface = uv.x * uv.x + uv.y * uv.y;

	// Keep a minimum screen-space footprint. FilterSize is sqrt(2) / 2,
	// hence its inverse squared value is 2.
	const float2 d = {
		point_image.x - pixf.x,
		point_image.y - pixf.y
	};

	// low pass filter
	const float filter_inv_square = 1.0f / (FilterSize * FilterSize);
	const float rho_filter = filter_inv_square * (d.x * d.x + d.y * d.y);
	const float rho = min(rho_surface, rho_filter);

	eval.power = -0.5f * rho; // calculate the weight
	eval.depth = uv.x * Tw.x + uv.y * Tw.y + Tw.z;
	eval.valid = eval.power <= 0.0f && eval.depth > 0.2f;
	return eval;
}

/**
 * @brief: hybrid render CUDA kernel
 */
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	// --- tile 排序结果 ---
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,	
	// --- 图像信息 ---
	int W, int H,
	// --- 两种 Gaussian 共用数据 ---
	const uint8_t* __restrict__ gaussian_type,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	// --- 3D Gaussian ---
	const float4* __restrict__ conic_opacity,
	// ---2D Gaussian ---
	const float* __restrict__ transMats,
	const float4* __restrict__ normal_opacity,
	// --- 图像输出与反向传播状态 ---
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ out_invdepth)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // 沿 y 轴方向有多少 block
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; 
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x; // 获得一维数组中的 id
	float2 pixf = { (float)pix.x, (float)pix.y};
		
	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);	
	int toDo = range.y - range.x;
	
	// shared memory
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ uint8_t collected_type[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];	
	__shared__ float3 collected_Tu[BLOCK_SIZE];  
	__shared__ float3 collected_Tv[BLOCK_SIZE];  
	__shared__ float3 collected_Tw[BLOCK_SIZE];  

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float expected_invdepth = 0.0f;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			const int thread_idx = block.thread_rank();
			const uint8_t type = gaussian_type[coll_id];
			collected_id[thread_idx] = coll_id;
			collected_type[thread_idx] = type;
			collected_xy[thread_idx] = points_xy_image[coll_id];

			if (type == GAUSSIAN_2D)
			{
				collected_normal_opacity[thread_idx] = normal_opacity[coll_id];
				collected_Tu[thread_idx] = {transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
				collected_Tv[thread_idx] = {transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
				collected_Tw[thread_idx] = {transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
			}
			else
			{
				collected_conic_opacity[thread_idx] = conic_opacity[coll_id];
			}
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;

			RenderEval eval;
			if (collected_type[j] == GAUSSIAN_2D)
			{
				eval = render_mode_2d(
					pixf,
					collected_xy[j],
					collected_Tu[j],
					collected_Tv[j],
					collected_Tw[j],
					collected_normal_opacity[j]);
			}
			else
			{
				eval = render_mode_3d(
					pixf,
					collected_xy[j],
					collected_conic_opacity[j],
					depths[collected_id[j]]);
			}

			if (!eval.valid)
				continue;

			// Both render modes return the Gaussian exponent and opacity;
			// compositing is shared so 2D and 3D Gaussians preserve the
			// common front-to-back order produced by binning.
			const float alpha = min(0.99f, eval.opacity * expf(eval.power));
			if (alpha < 1.0f / 255.0f)
				continue;

			const float test_T = T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			const float weight = alpha * T;
			const int gaussian_id = collected_id[j];
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[gaussian_id * CHANNELS + ch] * weight;

			if (out_invdepth != nullptr)
				expected_invdepth += (1.0f / eval.depth) * weight;

			T = test_T;
			last_contributor = contributor;
		}
	}

	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (out_invdepth != nullptr)
			out_invdepth[pix_id] = expected_invdepth;
	}

}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const uint8_t* gaussian_type,
	const float2* means2D,
	const float* colors,
	const float* depths,
	const float4* conic_opacity,
	const float* transMats,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_invdepth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		gaussian_type,
		means2D,
		colors,
		depths,
		conic_opacity,
		transMats,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_invdepth);
}


void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const uint8_t* gaussian_type,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	float* transMats,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	// 前一个参数表示将高斯分为多少块, 后一个参数表示一块中有多少线程
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		gaussian_type,
		shs,
		clamped,
		transMat_precomp,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		transMats,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
