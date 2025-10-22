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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const size_t P,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	glm::vec4* __restrict__ dL_dscales,
	const float* __restrict__ scales,
	const float lambda,
	glm::vec2* __restrict__ dL_dproj,
	float * __restrict__ dL_dtransMat2)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float4 collected_scales[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	float depth_pre = 0.0f;
	float T_tmp = 1.0f;
	float T_last = final_Ts[pix_id + H*W*3];

	// We start from the back. The ID of the last contributing
	// Primitive is known from each pixel from the forward.
	uint32_t contributor = toDo;
	uint32_t testT_count = 0;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
	int used_contributor_list_last_idx = inside ? n_contrib[pix_id + H * W * 2] : 0;

#if RENDER_AXUTILITY
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];
		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};

	const float final_A = 1 - T_final;
#endif

	float last_alpha = 0;

	// Traverse all Primitives
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			collected_scales[block.thread_rank()] = {scales[4 * coll_id+0], scales[4 * coll_id+1], scales[4 * coll_id+2], scales[4 * coll_id+3]};
			
		}
		block.sync();

		// Iterate over Primitives
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{	
			// Keep track of current Primitive ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor) continue;

			int used_contributor = n_contrib[pix_id + H*W*(3+used_contributor_list_last_idx-1)];
			if (used_contributor != contributor) continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			const float4 Tscale = collected_scales[j];
			const float3 k = pix.x * Tw - Tu;
			const float3 l = pix.y * Tw - Tv;
			const float3 p = cross(k, l);
			if (p.z >= 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 

			float c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z; 

			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};

			// -------------------------------------------
			float px = s.x * Tscale.x;
			float py = s.y * Tscale.y;
			float wx = (Tscale.x - abs(px)) * lambda;
			float awx = (wx >= 0.0f) ? 1.0f : 2.0f*(1.0f / (1.0f + exp(-wx)));
			float wy = (Tscale.y - abs(py)) * lambda;
			float awy = (wy >= 0.0f) ? 1.0f : 2.0f*(1.0f / (1.0f + exp(-wy)));
			float beta = (awx < awy) ? 1.0f : 0.0f;
			const float G = beta * awx + (1.0f - beta) * awy;
			const float alpha = min(1.0f, G);
			// ------------------------------------------
			if (alpha < 0.0001f) continue;
			if (contributor == last_contributor-1)
			{
				T = T_last;
			}
			else
			{
				T = T / (1.f - alpha);
			}

			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;
			// Propagate gradients to per-Primitive colors and keep
			// gradients w.r.t. alpha (blending factor for a Primitive/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];

			float dL_dz = 0.0f;
			float dL_dweight = 0.0f;
#if RENDER_AXUTILITY
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
			}
			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Primitive normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Helpful reusable temporary variables
			const float dL_dG = dL_dalpha;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth; 
#endif

			// Update gradients w.r.t. covariance of Primitive 3x3 (T)
			const float dwx_dscalex = lambda;
			const float dwy_dscaley = lambda;
			// ------------------------------------------
			float dL_dsx = dL_dz * Tw.x;
			float dL_dsx2;
			float dL_dscalex_p;
			float dL_dscalex_n;
			float dL_dpx=0.0f;
			if (wx >= 0.0f || beta == 0.f){
				dL_dsx2 = dL_dz * Tw.x;
				dL_dscalex_p = 0.0f;
				dL_dscalex_n = 0.0f;
			} else{
				const float dawx_dwx = awx * (1.0f - awx / 2.0f);
				float dwx_dprojx_tmp = 0.0f;
				if (s.x > 0){
					dwx_dprojx_tmp = -lambda;
				}
				else if (s.x == 0){
					dwx_dprojx_tmp = 0.0f;
				}
				else if (s.x < 0)
				{
					dwx_dprojx_tmp = lambda;
				}
				const float dwx_dprojx = dwx_dprojx_tmp;

				dL_dsx2 = dL_dG * beta * dawx_dwx * dwx_dprojx * Tscale.x + dL_dz * Tw.x;
				dL_dpx = dL_dG * beta * dawx_dwx * dwx_dprojx;
				if (px >= 0)
				{
					dL_dscalex_p = dL_dG * beta * dawx_dwx * dwx_dscalex;
					dL_dscalex_n = 0.0f;
				}else
				{
					dL_dscalex_p = 0.0f;
					dL_dscalex_n = dL_dG * beta * dawx_dwx * dwx_dscalex;
				}
			}
			// ------------------------------------------
			float dL_dsy = dL_dz * Tw.y;
			float dL_dsy2;
			float dL_dscaley_p;
			float dL_dscaley_n;
			float dL_dpy = 0.0f;
			if (wy >= 0.0f || beta == 1.0f){
				dL_dsy2 = dL_dz * Tw.y;
				dL_dscaley_p = 0.f;
				dL_dscaley_n = 0.f;
			} else{
				const float dawy_dwy = awy * (1.0f - awy / 2.0f);
				float dwy_dprojy_tmp = 0.0f;
				if (s.y > 0){
					dwy_dprojy_tmp = -lambda;
				}
				else if (s.y == 0){
					dwy_dprojy_tmp = 0.0f;
				}
				else if (s.y < 0)
				{
					dwy_dprojy_tmp = lambda;
				}
				const float dwy_dprojy = dwy_dprojy_tmp;
				dL_dsy2 = dL_dG * (1.0f - beta) * dawy_dwy * dwy_dprojy * Tscale.y + dL_dz * Tw.y;
				dL_dpy = dL_dG * (1.0f - beta) * dawy_dwy * dwy_dprojy;
				if (py >= 0)
				{
					dL_dscaley_p = dL_dG * (1.0f - beta) * dawy_dwy * dwy_dscaley;
					dL_dscaley_n = 0.f;
				}else
				{
					dL_dscaley_p = 0.f;
					dL_dscaley_n = dL_dG * (1.0f - beta) * dawy_dwy * dwy_dscaley;
				}
				
			}
			// ------------------------------------------
			const float2 dL_ds = {
					dL_dsx,
					dL_dsy
				};
			const float2 dL_ds2 = {
					dL_dsx2,
					dL_dsy2
				};
			const float4 dL_dscale = {
					dL_dscalex_p,
					dL_dscaley_p,
					dL_dscalex_n,
					dL_dscaley_n
				};
			// ------------------------------------------
			const float3 dz_dTw = {s.x, s.y, 1.0};
			const float dsx_pz = dL_ds.x / p.z;
			const float dsy_pz = dL_ds.y / p.z;
			const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
			const float3 dL_dk = cross(l, dL_dp);
			const float3 dL_dl = cross(dL_dp, k);
			const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
			const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
			const float3 dL_dTw = {
				pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
				pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
				pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};

			// Update gradients w.r.t. 3D covariance (3x3 matrix)
			atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
			atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
			atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
			atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
			atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
			atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
			atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
			atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
			atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);

			const float dsx_pz2 = dL_ds2.x / p.z;
			const float dsy_pz2 = dL_ds2.y / p.z;
			const float3 dL_dp2 = {dsx_pz2, dsy_pz2, -(dsx_pz2 * s.x + dsy_pz2 * s.y)};
			const float3 dL_dk2 = cross(l, dL_dp2);
			const float3 dL_dl2 = cross(dL_dp2, k);
			const float3 dL_dTu2 = {-dL_dk2.x, -dL_dk2.y, -dL_dk2.z};
			const float3 dL_dTv2 = {-dL_dl2.x, -dL_dl2.y, -dL_dl2.z};
			const float3 dL_dTw2 = {
				pixf.x * dL_dk2.x + pixf.y * dL_dl2.x + dL_dz * dz_dTw.x, 
				pixf.x * dL_dk2.y + pixf.y * dL_dl2.y + dL_dz * dz_dTw.y, 
				pixf.x * dL_dk2.z + pixf.y * dL_dl2.z + dL_dz * dz_dTw.z};
			
			atomicAdd(&dL_dtransMat2[global_id * 9 + 0],  dL_dTu2.x);
			atomicAdd(&dL_dtransMat2[global_id * 9 + 1],  dL_dTu2.y);
			atomicAdd(&dL_dtransMat2[global_id * 9 + 2],  dL_dTu2.z);
			atomicAdd(&dL_dtransMat2[global_id * 9 + 3],  dL_dTv2.x);
			atomicAdd(&dL_dtransMat2[global_id * 9 + 4],  dL_dTv2.y);
			atomicAdd(&dL_dtransMat2[global_id * 9 + 5],  dL_dTv2.z);
			atomicAdd(&dL_dtransMat2[global_id * 9 + 6],  dL_dTw2.x);
			atomicAdd(&dL_dtransMat2[global_id * 9 + 7],  dL_dTw2.y);
			atomicAdd(&dL_dtransMat2[global_id * 9 + 8],  dL_dTw2.z);

			atomicAdd(&(dL_dscales[global_id]).x, dL_dscale.x);
			atomicAdd(&(dL_dscales[global_id]).y, dL_dscale.y);
			atomicAdd(&(dL_dscales[global_id]).z, dL_dscale.z);
			atomicAdd(&(dL_dscales[global_id]).w, dL_dscale.w);

			atomicAdd(&(dL_dproj[global_id]).x, dL_dpx);
			atomicAdd(&(dL_dproj[global_id]).y, dL_dpy);
			
			used_contributor_list_last_idx--;
		}
	}
}


__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec2* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float3* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec4* dL_drots,
	const glm::vec2* dL_dproj,
	float* dL_dTs2,
	const float* image_center)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Primitive
	if (false) 
	{
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
		rot = rots[idx];
		R = quat_to_rotmat(rot);
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);
		
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
			glm::vec4(L[0], 0.0),
			glm::vec4(L[1], 0.0),
			glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
		);

		glm::mat4 world2ndc = glm::mat4(
			projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
			projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
			projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
			projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
		);

		glm::mat3x4 ndc2pix = glm::mat3x4(
			glm::vec4(float(W) / 2.0, 0.0, 0.0, image_center[0]),
			glm::vec4(0.0, float(H) / 2.0, 0.0, image_center[1]),
			glm::vec4(0.0, 0.0, 0.0, 1.0)
		);

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Primitive
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	glm::mat3 dL_dT2 = glm::mat3(
		dL_dTs2[idx*9+0], dL_dTs2[idx*9+1], dL_dTs2[idx*9+2],
		dL_dTs2[idx*9+3], dL_dTs2[idx*9+4], dL_dTs2[idx*9+5],
		dL_dTs2[idx*9+6], dL_dTs2[idx*9+7], dL_dTs2[idx*9+8]
	);

	if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Primitive
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	glm::mat3x4 dL_dM2 = P * glm::transpose(dL_dT2);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);  // equal to grad of stK_inter_n.sum()
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS2 = glm::mat3(
		glm::vec3(dL_dM2[0]),
		glm::vec3(dL_dM2[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);
	glm::mat3 dL_dR2 = glm::mat3(
		dL_dRS2[0] * glm::vec3(scale.x),
		dL_dRS2[1] * glm::vec3(scale.y),
		dL_dRS2[2]);
	
	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR2);
	float mx = -dL_dproj[idx].x * R[0][0] - dL_dproj[idx].y * R[1][0];
	float my = -dL_dproj[idx].x * R[0][1] - dL_dproj[idx].y * R[1][1];
	float mz = -dL_dproj[idx].x * R[0][2] - dL_dproj[idx].y * R[1][2];
	dL_dmeans[idx] = glm::vec3(dL_dM[2]) + glm::vec3(mx, my, mz);
}

template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	float* dL_dtransMats,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	float3* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec4* dL_drots,
	glm::vec2* dL_dproj,
	float* dL_dtransMats2,
	const float* image_center)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_drots,
		dL_dproj,
		(dL_dtransMats2),
		image_center
	);

	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec2* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos, 
	float3* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec4* dL_drots,
	glm::vec2* dL_dproj,
	float* dL_dtransMats2,
	const float* image_center)
{	
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_drots,
		dL_dproj,
		dL_dtransMats2,
		image_center
	);
}

void BACKWARD::render(
	const size_t P,
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* bg_color,
	const float2* means2D,
	const float4* normal_opacity,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_depths,
	float * dL_dtransMat,
	float3* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors,
	glm::vec4* dL_dscales,
	const float* scales,
	const float lambda,
	glm::vec2* dL_dproj,
	float * dL_dtransMat2)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		P,
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		bg_color,
		means2D,
		normal_opacity,
		transMats,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors,
		dL_dscales,
		scales,
		lambda,
		dL_dproj,
		dL_dtransMat2);
}
