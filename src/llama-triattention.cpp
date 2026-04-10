// TriAttention: Trigonometric KV Cache Eviction for llama.cpp
// Based on arXiv 2604.04921 (MIT/NVIDIA/ZJU)
//
// This file implements the complete TriAttention scoring and pruning pipeline:
//   1. Binary calibration file loader (.triattention format)
//   2. RoPE inversion (post-RoPE K → pre-RoPE K)
//   3. Trigonometric key importance scoring (Eqs. 6-10 from paper)
//   4. Three pruning modes: global union, per-KV-head, per-layer-per-head
//   5. Position tracking hooks for correct RoPE inversion after pruning
//   6. KV cache integration hooks
//
// All math references cite equation numbers from: "TriAttention: Decoding-Time
// Trigonometric Key Cache Eviction for Long-Context LLM Inference" (2604.04921)

#include "llama-triattention.h"
#include "llama-triattention-file.h"
#include "llama-kv-cache.h"
#include "llama-hparams.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"   // GPU scoring: triattention_gpu_init, _score_head, etc.

// Block types and dequant declarations are in ggml-common.h (ggml/src/)
// which is not on the include path for src/. We declare the dequant
// functions with void* parameters and cast at call sites.
// Block sizes (bytes per 128 elements): turbo2=10, turbo3=14, turbo4=68, q8_0=34

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// For timing
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Pre-computed WHT inverse rotation matrix R^T (128x128)
// Used to convert turbo2/turbo3 dequant output from WHT-rotated space
// back to the original post-RoPE embedding space.
// turbo4 dequant already applies R^T internally, so this is only needed
// for turbo2_0 and turbo3_0 types.
#include "turbo-rotation-data.h"

// TurboQuant dequant function declarations (from ggml-turbo-quant.c)
// Using void* since block type definitions live in ggml-common.h (not on include path)
extern "C" {
    void dequantize_row_turbo2_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    void dequantize_row_turbo3_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    void dequantize_row_turbo4_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
}

// Standard ggml dequant for Q8_0, F16, etc.
extern "C" {
    void dequantize_row_q8_0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
}

#if !defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
extern "C" {
    triattention_gpu_state * triattention_gpu_init(
        const struct triattention_gpu_config *,
        const struct triattention_gpu_head_calib *,
        const float *,
        const float *,
        const float *,
        void *) {
        return nullptr;
    }

    void triattention_gpu_score_head(
        triattention_gpu_state *,
        const void *,
        uint64_t,
        size_t,
        uint32_t,
        uint32_t,
        const uint32_t *,
        const int32_t *,
        uint32_t,
        int64_t,
        int,
        float *,
        void *) {
    }

    void triattention_gpu_scores_to_host(float *, const float *, uint32_t, void *) {
    }

    void triattention_gpu_upload_cells(
        uint32_t **,
        int32_t **,
        const uint32_t *,
        const int32_t *,
        uint32_t,
        void *) {
    }

    float * triattention_gpu_alloc_scores(uint32_t, void *) {
        return nullptr;
    }

    void triattention_gpu_free_dev(void *) {
    }

    void triattention_gpu_free(triattention_gpu_state *) {
    }
}
#endif

// ============================================================================
// Internal helpers
// ============================================================================

static double triattention_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

static void minmax_normalize(float * scores, uint32_t n);

// Matrix-vector multiply: out[i] = sum_j mat[i*d + j] * vec[j]
// Used for inverse WHT rotation on turbo2/turbo3 dequant output
static void matvec_128(const float * mat, const float * vec, float * out) {
    for (int i = 0; i < 128; i++) {
        float sum = 0.0f;
        const float * row = mat + i * 128;
        for (int j = 0; j < 128; j++) {
            sum += row[j] * vec[j];
        }
        out[i] = sum;
    }
}

// Build geometric offset array: {1, 2, 4, 8, ..., offset_max}
// Paper Eq. 9: D = {2^0, 2^1, ..., 2^{log2(max_length)}}
static uint32_t triattention_build_offsets(float * offsets, uint32_t offset_max) {
    uint32_t n = 0;
    for (uint32_t d = 1; d <= offset_max; d *= 2) {
        offsets[n++] = (float)d;
    }
    return n;
}

// Precompute derived quantities per head from calibration stats:
//   q_mean_abs[f] = sqrt(q_mean_real[f]^2 + q_mean_imag[f]^2)  = ||E[q_f]||
//   extra_weight[f] = q_abs_mean[f] - q_mean_abs[f]              = E[||q_f||] - ||E[q_f]||
// Paper Eq. 8: the "norm excess" term weighted by (1 - R_f)
static void triattention_precompute_head_derived(triattention_head_stats * hs, uint32_t freq_count, bool disable_mlr) {
    hs->q_mean_abs   = new float[freq_count];
    hs->extra_weight = new float[freq_count];

    for (uint32_t f = 0; f < freq_count; f++) {
        float re = hs->q_mean_real[f];
        float im = hs->q_mean_imag[f];
        hs->q_mean_abs[f] = sqrtf(re * re + im * im);

        if (disable_mlr) {
            // Ablation: use q_abs_mean directly as the norm contribution
            hs->extra_weight[f] = hs->q_abs_mean[f];
        } else {
            // Standard: MLR-weighted norm excess = E[||q_f||] - ||E[q_f]||
            // This is >= 0 because ||E[x]|| <= E[||x||] (Jensen's inequality)
            hs->extra_weight[f] = hs->q_abs_mean[f] - hs->q_mean_abs[f];
            if (hs->extra_weight[f] < 0.0f) {
                hs->extra_weight[f] = 0.0f;  // Numerical safety
            }
        }
    }
}

// ============================================================================
// Core scoring functions: CPU implementations
// ============================================================================

// Invert RoPE rotation on post-RoPE key vectors.
// Paper Eq. 4: recover pre-RoPE K from post-RoPE K using known positions.
//
// For "half" style (Llama/Qwen): dimensions split as [real | imag]
//   k_pre[f]    = k_post[f]*cos(omega[f]*pos) + k_post[f+fc]*sin(omega[f]*pos)
//   k_pre[f+fc] = k_post[f+fc]*cos(omega[f]*pos) - k_post[f]*sin(omega[f]*pos)
//
// For "interleaved" style: pairs are (2f, 2f+1)
//   k_pre[2f]   = k_post[2f]*cos(omega[f]*pos) + k_post[2f+1]*sin(omega[f]*pos)
//   k_pre[2f+1] = k_post[2f+1]*cos(omega[f]*pos) - k_post[2f]*sin(omega[f]*pos)
void triattention_invert_rope(
    float       * out,
    const float * post_rope_k,
    const int32_t * positions,
    const float * omega,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t rope_style)
{
    for (uint32_t i = 0; i < n_keys; i++) {
        const float * src = post_rope_k + (size_t)i * head_dim;
        float       * dst = out         + (size_t)i * head_dim;
        const float   pos = (float)positions[i];

        if (rope_style == 0) {
            // Half style: [real_0..real_{fc-1} | imag_0..imag_{fc-1}]
            for (uint32_t f = 0; f < freq_count; f++) {
                float angle = omega[f] * pos;
                float c = cosf(angle);
                float s = sinf(angle);
                float re = src[f];
                float im = src[f + freq_count];
                // Invert rotation: multiply by conjugate rotation matrix
                dst[f]              = re * c + im * s;
                dst[f + freq_count] = im * c - re * s;
            }
        } else {
            // Interleaved source layout: [re_0, im_0, re_1, im_1, ...]
            // Output is converted to the shared "half" layout expected by the scorer:
            // [real_0..real_{fc-1} | imag_0..imag_{fc-1}]
            for (uint32_t f = 0; f < freq_count; f++) {
                float angle = omega[f] * pos;
                float c = cosf(angle);
                float s = sinf(angle);
                float re = src[2 * f];
                float im = src[2 * f + 1];
                dst[f]              = re * c + im * s;
                dst[f + freq_count] = im * c - re * s;
            }
        }
    }
}

// Score cached keys for a single (layer, attention_head) pair.
// Paper Eqs. 6-10: trigonometric importance scoring with MLR norm term.
//
// For each key at position p_k with base distance Delta = round_start - p_k:
//   1. Convert pre-RoPE K to complex representation
//   2. Compute amplitude: amp_f = ||E[q_f]|| * |k_f|
//   3. Compute phase: phi_f = angle(E[q_f] * conj(k_f))
//   4. Compute trig score: S_trig(Delta+delta) = sum_f amp_f * fscale_sq_f * cos(omega_f*(Delta+delta) + phi_f)
//   5. Compute norm score: S_norm = sum_f extra_f * fscale_sq_f * |k_f|
//   6. Aggregate over geometric offsets
void triattention_score_keys(
    float       * out_scores,
    const float * pre_rope_k,
    const triattention_head_stats * stats,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets,
    const int32_t * key_positions,
    int64_t  round_start,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t n_offsets,
    enum triattention_agg agg,
    bool disable_trig)
{
    const float inv_n_offsets = 1.0f / (float)n_offsets;

    for (uint32_t i = 0; i < n_keys; i++) {
        const float * k = pre_rope_k + (size_t)i * head_dim;
        const float   base_delta = (float)(round_start - key_positions[i]);

        // Precompute per-frequency quantities for this key
        // Using "half" layout: k_re = k[f], k_im = k[f + freq_count]
        // (interleaved would be k[2f], k[2f+1] — handled at invert_rope stage,
        //  output from invert_rope is always in half layout for scoring)

        float total_score = 0.0f;

        if (!disable_trig) {
            // Full scoring: trigonometric + norm terms
            for (uint32_t d = 0; d < n_offsets; d++) {
                float delta = base_delta + offsets[d];
                float offset_score = 0.0f;

                for (uint32_t f = 0; f < freq_count; f++) {
                    float k_re = k[f];
                    float k_im = k[f + freq_count];
                    float k_mag = sqrtf(k_re * k_re + k_im * k_im);

                    // Amplitude: ||E[q_f]|| * |k_f|  (Paper Eq. 7)
                    float amp = stats->q_mean_abs[f] * k_mag;

                    // Phase from conj multiply: E[q_f] * conj(k_f)
                    // = (q_re + i*q_im) * (k_re - i*k_im)
                    // = (q_re*k_re + q_im*k_im) + i*(q_im*k_re - q_re*k_im)
                    float conj_re = stats->q_mean_real[f] * k_re + stats->q_mean_imag[f] * k_im;
                    float conj_im = stats->q_mean_imag[f] * k_re - stats->q_mean_real[f] * k_im;
                    float phi = atan2f(conj_im, conj_re);

                    // Trigonometric score (Paper Eq. 6):
                    // S_trig += amp * fscale^2 * cos(omega * delta + phi)
                    float phase = omega[f] * delta + phi;
                    offset_score += amp * freq_scale_sq[f] * cosf(phase);

                    // Norm excess term (Paper Eq. 8):
                    // S_norm += extra_weight * fscale^2 * |k_f|
                    offset_score += stats->extra_weight[f] * freq_scale_sq[f] * k_mag;
                }

                if (agg == TRIATTENTION_AGG_MAX) {
                    total_score = (d == 0) ? offset_score : fmaxf(total_score, offset_score);
                } else {
                    total_score += offset_score;
                }
            }

            if (agg == TRIATTENTION_AGG_MEAN) {
                total_score *= inv_n_offsets;
            }
        } else {
            // Ablation: norm-only scoring (disable_trig=true)
            // Only the position-independent norm term
            for (uint32_t f = 0; f < freq_count; f++) {
                float k_re = k[f];
                float k_im = k[f + freq_count];
                float k_mag = sqrtf(k_re * k_re + k_im * k_im);
                total_score += stats->extra_weight[f] * freq_scale_sq[f] * k_mag;
            }
        }

        out_scores[i] = total_score;
    }
}

void triattention_score_keys_norm(
    float       * out_scores,
    const float * pre_rope_k,
    const float * freq_scale_sq,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count) {
    for (uint32_t i = 0; i < n_keys; ++i) {
        const float * k = pre_rope_k + (size_t) i * head_dim;
        float total = 0.0f;
        for (uint32_t f = 0; f < freq_count; ++f) {
            const float re = k[f];
            const float im = k[f + freq_count];
            const float mag = sqrtf(re * re + im * im);
            total += freq_scale_sq[f] * mag;
        }
        out_scores[i] = total;
    }
}

void triattention_build_recency_scores(
    float         * out_scores,
    const int32_t * key_positions,
    uint32_t        n_keys) {
    if (!out_scores || !key_positions || n_keys == 0) {
        return;
    }

    int32_t min_pos = key_positions[0];
    int32_t max_pos = key_positions[0];
    for (uint32_t i = 1; i < n_keys; ++i) {
        min_pos = std::min(min_pos, key_positions[i]);
        max_pos = std::max(max_pos, key_positions[i]);
    }

    const float max_age = (float) std::max<int32_t>(1, max_pos - min_pos);
    for (uint32_t i = 0; i < n_keys; ++i) {
        const float age = (float) (max_pos - key_positions[i]);
        out_scores[i] = 1.0f - age / max_age;
    }
}

void triattention_blend_fallback_scores(
    float       * scores,
    const float * recency_scores,
    uint32_t      n_keys,
    float         lambda) {
    if (!scores || !recency_scores || n_keys == 0) {
        return;
    }

    lambda = fminf(1.0f, fmaxf(0.0f, lambda));
    minmax_normalize(scores, n_keys);
    for (uint32_t i = 0; i < n_keys; ++i) {
        scores[i] = (1.0f - lambda) * scores[i] + lambda * recency_scores[i];
    }
}

// ============================================================================
// KV cache dequantization helper
// ============================================================================

// Dequantize K values for a specific KV head from the cache tensor.
// Handles all supported quantization types and applies inverse WHT
// rotation for turbo2/turbo3 types.
//
// Parameters:
//   out         — [n_cells, padded_head_dim] dequantized float output
//   k_tensor    — the raw K cache tensor for this layer
//   cell_indices— [n_cells] which cell slots to extract
//   kv_head_idx — which KV head (0..n_kv_heads-1)
//   n_cells     — number of cells to dequantize
//   padded_hd   — padded head dimension (128-aligned for turbo types)
//   n_kv_heads  — total number of KV heads
//   need_wht_inv— whether to apply inverse WHT rotation (turbo2/turbo3)
//
// Note: This function copies data from potentially GPU-resident tensors
// to CPU memory, which involves a synchronous transfer. This is acceptable
// because pruning happens infrequently (every divide_length tokens).
static void triattention_dequant_kv_head(
    float              * out,
    const ggml_tensor  * k_tensor,
    const uint32_t     * cell_indices,
    uint32_t             kv_head_idx,
    uint32_t             n_cells,
    uint32_t             padded_hd,
    bool                 need_wht_inv)
{
    const ggml_type k_type = k_tensor->type;
    const uint64_t  n_embd_k_gqa = k_tensor->ne[0];  // total K embedding (all KV heads)
    const size_t    row_bytes = ggml_row_size(k_type, n_embd_k_gqa);

    // Byte offset to this KV head within a row
    const size_t head_offset_bytes = ggml_row_size(k_type, (uint64_t)kv_head_idx * padded_hd);
    const size_t head_bytes = ggml_row_size(k_type, padded_hd);

    // Temporary buffer for one quantized head block
    std::vector<uint8_t> quant_buf(head_bytes);

    // Temporary buffer for dequantized values (before WHT inverse)
    std::vector<float> dequant_tmp(padded_hd);

    for (uint32_t ci = 0; ci < n_cells; ci++) {
        const uint32_t cell_idx = cell_indices[ci];

        // Byte offset in the full tensor: row_bytes * cell_idx + head_offset_bytes
        // This addresses stream 0 (the common case for unified KV caches)
        const size_t tensor_offset = (size_t)cell_idx * row_bytes + head_offset_bytes;

        // Copy quantized data from backend (may be GPU) to CPU
        ggml_backend_tensor_get(k_tensor, quant_buf.data(), tensor_offset, head_bytes);

        // Dequantize based on type
        float * dst = need_wht_inv ? dequant_tmp.data() : (out + (size_t)ci * padded_hd);

        switch (k_type) {
            case GGML_TYPE_TURBO3_0:
                dequantize_row_turbo3_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_TURBO4_0:
                dequantize_row_turbo4_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_TURBO2_0:
                dequantize_row_turbo2_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_Q8_0:
                dequantize_row_q8_0(quant_buf.data(), dst, padded_hd);
                break;
            case GGML_TYPE_F16: {
                const ggml_fp16_t * src16 = (const ggml_fp16_t *)quant_buf.data();
                for (uint32_t j = 0; j < padded_hd; j++) {
                    dst[j] = ggml_fp16_to_fp32(src16[j]);
                }
                break;
            }
            case GGML_TYPE_BF16: {
                const ggml_bf16_t * src16 = (const ggml_bf16_t *)quant_buf.data();
                for (uint32_t j = 0; j < padded_hd; j++) {
                    dst[j] = ggml_bf16_to_fp32(src16[j]);
                }
                break;
            }
            case GGML_TYPE_F32: {
                memcpy(dst, quant_buf.data(), padded_hd * sizeof(float));
                break;
            }
            default:
                fprintf(stderr, "[TriAttention] ERROR: unsupported K cache type %d\n", k_type);
                memset(out + (size_t)ci * padded_hd, 0, padded_hd * sizeof(float));
                continue;
        }

        // Apply inverse WHT rotation for turbo2/turbo3
        // turbo4 dequant already applies R^T internally
        if (need_wht_inv) {
            float * final_dst = out + (size_t)ci * padded_hd;
            // Process in 128-element blocks (WHT block size)
            for (uint32_t b = 0; b < padded_hd; b += 128) {
                matvec_128(TURBO_ROTATION_RT, dequant_tmp.data() + b, final_dst + b);
            }
        }
    }
}

static uint32_t triattention_padded_dim(uint32_t head_dim) {
    return ((head_dim + 127u) / 128u) * 128u;
}

static void triattention_extract_rope_slice(
    float       * out,
    const float * full_head,
    uint32_t      n_keys,
    uint32_t      full_head_dim,
    uint32_t      rope_offset,
    uint32_t      rope_dim) {
    for (uint32_t i = 0; i < n_keys; ++i) {
        memcpy(out + (size_t) i * rope_dim,
               full_head + (size_t) i * full_head_dim + rope_offset,
               sizeof(float) * rope_dim);
    }
}

// ============================================================================
// Public API: Init / Free
// ============================================================================

static triattention_state * triattention_init_internal(
    const triattention_calibration * full_cal,
    bool fallback_active,
    const char * source_name,
    const triattention_config * cfg,
    const triattention_model_params * model,
    const uint32_t * sampled_layers,
    uint32_t n_sampled_layers) {
    triattention_calibration * cal = nullptr;
    if (full_cal) {
        cal = triattention_calibration_subset(full_cal, sampled_layers, n_sampled_layers);
        if (!cal) {
            return nullptr;
        }
    } else {
        cal = triattention_calibration_create_fallback(model, sampled_layers, n_sampled_layers);
        if (!cal) {
            fprintf(stderr, "[TriAttention] ERROR: failed to construct runtime fallback state\n");
            return nullptr;
        }
        fallback_active = true;
    }

    const char * resolved_source = source_name && source_name[0] ? source_name : (fallback_active ? "runtime-fallback" : "<unknown>");

    // Allocate state
    auto * state = new triattention_state();
    memset(state, 0, sizeof(triattention_state));

    state->cal = cal;
    state->cfg = *cfg;
    state->cfg.fallback_recency_weight = fminf(1.0f, fmaxf(0.0f, state->cfg.fallback_recency_weight));
    state->model = {};
    state->model.kv_size = model->kv_size;
    state->model.num_layers = model->num_layers;
    state->model.head_dim = model->head_dim;
    state->model.rope_dim = model->rope_dim;
    state->model.num_attn_heads = model->num_attn_heads;
    state->model.num_kv_heads = model->num_kv_heads;
    state->model.rope_style = model->rope_style;
    state->model.n_ctx_orig = model->n_ctx_orig;
    state->model.rope_theta = model->rope_theta;
    state->model.rope_freq_scale = model->rope_freq_scale;
    state->model.rope_ext_factor = model->rope_ext_factor;
    state->model.rope_attn_factor = model->rope_attn_factor;
    state->model.rope_beta_fast = model->rope_beta_fast;
    state->model.rope_beta_slow = model->rope_beta_slow;
    state->fallback_active = fallback_active;
    state->kv_size = model->kv_size;
    state->absolute_position = 0;
    state->prefix_length     = 0;

    for (uint32_t il = 0; il < cal->num_layers; ++il) {
        triattention_layer_params & cal_layer = cal->layers[il];
        const triattention_layer_params & model_layer = model->layers[il];
        cal_layer.kv_source_layer = model_layer.kv_source_layer;

        if ((!cal_layer.omega || !cal_layer.freq_scale_sq) && model_layer.omega && model_layer.freq_scale_sq) {
            delete[] cal_layer.omega;
            delete[] cal_layer.freq_scale_sq;
            cal_layer.omega = new float[model_layer.freq_count];
            cal_layer.freq_scale_sq = new float[model_layer.freq_count];
            memcpy(cal_layer.omega, model_layer.omega, sizeof(float) * model_layer.freq_count);
            memcpy(cal_layer.freq_scale_sq, model_layer.freq_scale_sq, sizeof(float) * model_layer.freq_count);
        }

        state->max_head_dim       = std::max(state->max_head_dim, cal_layer.head_dim);
        state->max_rope_dim       = std::max(state->max_rope_dim, cal_layer.rope_dim);
        state->max_freq_count     = std::max(state->max_freq_count, cal_layer.freq_count);
        state->max_padded_head_dim = std::max(state->max_padded_head_dim, triattention_padded_dim(cal_layer.head_dim));
    }

    state->offsets = new float[32];
    state->n_offsets = triattention_build_offsets(state->offsets, cfg->offset_max);

    if (!fallback_active) {
        for (uint32_t h = 0; h < cal->n_sampled; h++) {
            const uint32_t layer_idx = cal->sampled_layer[h];
            triattention_precompute_head_derived(&cal->head_stats[h], cal->layers[layer_idx].freq_count, cfg->disable_mlr);
        }
    }

    state->cell_positions = new int32_t[state->kv_size];
    for (uint32_t i = 0; i < state->kv_size; i++) {
        state->cell_positions[i] = -1;
    }

    state->dequant_buf  = new float[(size_t) state->kv_size * state->max_padded_head_dim];
    state->rope_buf     = new float[(size_t) state->kv_size * state->max_rope_dim];
    state->unrot_buf    = new float[(size_t) state->kv_size * state->max_rope_dim];
    state->score_buf    = new float[(size_t)cal->n_sampled * state->kv_size];
    state->combined_buf = new float[state->kv_size];
    state->keep_indices = new uint32_t[cfg->budget];

    state->total_prune_calls   = 0;
    state->total_tokens_evicted = 0;
    state->total_prune_time_ms = 0.0;
    state->last_prune_time_ms  = 0.0;

    fprintf(stderr, "[TriAttention] Initialized: %s, source=%s, budget=%u, window=%u, mode=%d, offsets=%u, "
            "kv_size=%u, sampled_heads=%u\n",
            fallback_active ? "experimental fallback" : "calibrated",
            resolved_source,
            cfg->budget, cfg->divide_length, (int)cfg->mode,
            state->n_offsets, state->kv_size, cal->n_sampled);

    return state;
}

triattention_state * triattention_init(
    const char * stats_path,
    const triattention_config * cfg,
    const triattention_model_params * model,
    const uint32_t * sampled_layers,
    uint32_t n_sampled_layers)
{
    if (!cfg || !model) {
        return nullptr;
    }

    triattention_calibration * full_cal = nullptr;
    bool fallback_active = false;

    if (stats_path && stats_path[0] != '\0') {
        full_cal = triattention_calibration_load(stats_path);
        if (!full_cal) {
            return nullptr;
        }
        if (!triattention_calibration_validate(full_cal, model, true)) {
            triattention_calibration_free(full_cal);
            return nullptr;
        }
    } else {
        if (cfg->fallback_mode == TRIATTENTION_FALLBACK_OFF) {
            fprintf(stderr, "[TriAttention] ERROR: no calibration file was provided and fallback is disabled\n");
            return nullptr;
        }
        if (cfg->fallback_mode != TRIATTENTION_FALLBACK_AUTO &&
            cfg->fallback_mode != TRIATTENTION_FALLBACK_HYBRID_NORM_RECENCY) {
            fprintf(stderr, "[TriAttention] ERROR: unsupported fallback mode %d\n", (int) cfg->fallback_mode);
            return nullptr;
        }
        fallback_active = true;
    }
    triattention_state * state = triattention_init_internal(full_cal, fallback_active, stats_path, cfg, model, sampled_layers, n_sampled_layers);
    triattention_calibration_free(full_cal);
    return state;
}

triattention_state * triattention_init_from_calibration(
    const triattention_calibration * calibration,
    const char * source_name,
    const triattention_config * cfg,
    const triattention_model_params * model,
    const uint32_t * sampled_layers,
    uint32_t n_sampled_layers) {
    if (!cfg || !model || !calibration) {
        return nullptr;
    }
    if (!triattention_calibration_validate(calibration, model, true)) {
        return nullptr;
    }
    return triattention_init_internal(calibration, false, source_name, cfg, model, sampled_layers, n_sampled_layers);
}

void triattention_free(triattention_state * state) {
    if (!state) return;

    triattention_calibration_free(state->cal);
    triattention_model_params_clear(&state->model);

    delete[] state->offsets;
    delete[] state->cell_positions;
    delete[] state->dequant_buf;
    delete[] state->rope_buf;
    delete[] state->unrot_buf;
    delete[] state->score_buf;
    delete[] state->combined_buf;
    delete[] state->keep_indices;

    // Free GPU scoring resources if initialized
    if (state->d_scores) {
        triattention_gpu_free_dev(state->d_scores);
        state->d_scores = nullptr;
    }
    if (state->d_gpu_state) {
        triattention_gpu_free((triattention_gpu_state *)state->d_gpu_state);
        state->d_gpu_state = nullptr;
    }

    delete state;
}

// ============================================================================
// Position tracking hooks
// ============================================================================

void triattention_on_token_added(
    triattention_state * state,
    uint32_t cell_idx,
    int32_t  abs_pos)
{
    if (!state || cell_idx >= state->kv_size) return;
    state->cell_positions[cell_idx] = abs_pos;
    if (abs_pos + 1 > (int32_t)state->absolute_position) {
        state->absolute_position = abs_pos + 1;
    }
}

void triattention_on_cell_removed(
    triattention_state * state,
    uint32_t cell_idx)
{
    if (!state || cell_idx >= state->kv_size) return;
    state->cell_positions[cell_idx] = -1;
}

void triattention_on_position_shift(
    triattention_state * state,
    int32_t delta,
    int32_t p0,
    int32_t p1)
{
    if (!state || delta == 0) return;

    for (uint32_t i = 0; i < state->kv_size; i++) {
        int32_t pos = state->cell_positions[i];
        if (pos >= 0 && pos >= p0 && (p1 < 0 || pos < p1)) {
            state->cell_positions[i] = pos + delta;
            if (state->cell_positions[i] < 0) {
                state->cell_positions[i] = -1;
            }
        }
    }
}

void triattention_on_reset(triattention_state * state) {
    if (!state) return;
    state->absolute_position = 0;
    state->prefix_length     = 0;
    for (uint32_t i = 0; i < state->kv_size; i++) {
        state->cell_positions[i] = -1;
    }
}

// ============================================================================
// Trigger logic
// ============================================================================

bool triattention_should_prune(
    const triattention_state * state,
    uint32_t n_used)
{
    if (!state) return false;

    switch (state->cfg.trigger) {
        case TRIATTENTION_TRIGGER_INTERVAL:
            return n_used >= state->cfg.budget &&
                   state->absolute_position > 0 &&
                   (state->absolute_position % state->cfg.divide_length) == 0;

        case TRIATTENTION_TRIGGER_SLACK:
            return n_used >= (state->cfg.budget + state->cfg.divide_length);

        default:
            return false;
    }
}

// ============================================================================
// Main pruning implementation
// ============================================================================

// Helper: z-score normalize an array in-place
// After normalization: mean=0, std=1
static void zscore_normalize(float * scores, uint32_t n) {
    if (n <= 1) return;

    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += scores[i];
    double mean = sum / n;

    double var_sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double d = scores[i] - mean;
        var_sum += d * d;
    }
    double std = sqrt(var_sum / n);
    if (std < 1e-10) std = 1e-10;

    for (uint32_t i = 0; i < n; i++) {
        scores[i] = (float)((scores[i] - mean) / std);
    }
}

static void minmax_normalize(float * scores, uint32_t n) {
    if (n == 0) {
        return;
    }

    float min_score = scores[0];
    float max_score = scores[0];
    for (uint32_t i = 1; i < n; ++i) {
        min_score = fminf(min_score, scores[i]);
        max_score = fmaxf(max_score, scores[i]);
    }

    const float denom = max_score - min_score;
    if (denom <= 1e-10f) {
        for (uint32_t i = 0; i < n; ++i) {
            scores[i] = 1.0f;
        }
        return;
    }

    for (uint32_t i = 0; i < n; ++i) {
        scores[i] = (scores[i] - min_score) / denom;
    }
}

// Helper: partial argsort — find top-K indices by score (descending)
// Returns indices of the K highest-scoring elements
static void top_k_indices(
    uint32_t       * out_indices,
    const float    * scores,
    uint32_t         n,
    uint32_t         k)
{
    if (k >= n) {
        // Keep all
        for (uint32_t i = 0; i < n; i++) out_indices[i] = i;
        return;
    }

    // Create index array and partial sort
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&scores](uint32_t a, uint32_t b) {
            return scores[a] > scores[b];  // descending
        });

    for (uint32_t i = 0; i < k; i++) {
        out_indices[i] = idx[i];
    }
}

// ============================================================================
// GPU scoring: lazy init
// ============================================================================

// Initializes GPU scoring state on first prune, using the K tensor type
// of the actual cache to select the right kernel variant.
// k_type must be a type supported by the GPU kernel (Q4_K, Q8_0, F16, F32,
// TURBO2_0, TURBO3_0, TURBO4_0). On failure, falls back silently to CPU.
static void triattention_init_gpu(triattention_state * state, ggml_type k_type) {
    if (state->gpu_init_tried) return;
    state->gpu_init_tried = true;
    if (state->fallback_active) {
        return;
    }

    const triattention_calibration * cal = state->cal;
    const triattention_config & cfg = state->cfg;
    if (!cal || cal->num_layers == 0 || cal->heterogeneous_layout) {
        return;
    }

    const triattention_layer_params & layer0 = cal->layers[0];
    if (layer0.rope_offset != 0 || !layer0.omega || !layer0.freq_scale_sq) {
        return;
    }

    triattention_gpu_config gcfg = {};
    gcfg.head_dim     = layer0.rope_dim;
    gcfg.freq_count   = layer0.freq_count;
    gcfg.n_kv_heads   = layer0.num_kv_heads;
    gcfg.n_sampled    = cal->n_sampled;
    gcfg.n_offsets    = state->n_offsets;
    gcfg.k_type       = k_type;
    gcfg.need_wht_inv = (k_type == GGML_TYPE_TURBO2_0 || k_type == GGML_TYPE_TURBO3_0);
    gcfg.disable_trig = cfg.disable_trig;

    std::vector<triattention_gpu_head_calib> gcalibs(cal->n_sampled);
    for (uint32_t h = 0; h < cal->n_sampled; h++) {
        gcalibs[h].q_mean_real  = cal->head_stats[h].q_mean_real;
        gcalibs[h].q_mean_imag  = cal->head_stats[h].q_mean_imag;
        gcalibs[h].q_mean_abs   = cal->head_stats[h].q_mean_abs;
        gcalibs[h].extra_weight = cal->head_stats[h].extra_weight;
    }

    auto * gpu_st = triattention_gpu_init(
        &gcfg, gcalibs.data(),
        layer0.omega, layer0.freq_scale_sq, state->offsets, nullptr);
    if (!gpu_st) {
        fprintf(stderr, "[TriAttention] GPU init failed, using CPU scoring\n");
        return;
    }

    // Pre-allocate one head's worth of score buffer on device
    float * d_s = triattention_gpu_alloc_scores(state->kv_size, nullptr);
    if (!d_s) {
        triattention_gpu_free(gpu_st);
        fprintf(stderr, "[TriAttention] GPU score buffer alloc failed, using CPU scoring\n");
        return;
    }

    state->d_gpu_state  = gpu_st;
    state->d_scores     = d_s;
    state->use_gpu      = true;

    fprintf(stderr, "[TriAttention] GPU scoring enabled (k_type=%d, heads=%u)\n",
            (int)k_type, cal->n_sampled);
}

int32_t triattention_prune(
    triattention_state * state,
    llama_kv_cache     * kv)
{
    GGML_UNUSED(kv);
    return state ? 0 : -1;
}

// ============================================================================
// Internal pruning implementation (called from KV cache integration)
// ============================================================================

// This is the real workhorse. Called from llama_kv_cache::triattention_try_prune()
// with direct access to the K tensors.
//
// Parameters:
//   state       — TriAttention runtime state
//   k_tensors   — array of K cache tensors, indexed by internal layer id
//   n_layers    — number of layers in k_tensors array
//   layer_map   — maps model layer index → internal layer id in k_tensors
//   v_cells     — reference to the cache's cell metadata (for rm operations)
//   v_heads     — reference to the cache's head pointer array (for updating after prune)
//   kv_size     — cache capacity
//
// Returns: number of cells evicted
int32_t triattention_prune_impl(
    triattention_state * state,
    ggml_tensor * const * k_tensors,
    uint32_t              n_layers,
    const int32_t       * layer_to_cache,
    uint32_t              kv_size)
{
    if (!state) return -1;

    double t_start = triattention_time_ms();

    const auto & cfg = state->cfg;
    const auto * cal = state->cal;
    const uint32_t budget = cfg.budget;

    // ---- Step 1: Enumerate occupied cells ----
    std::vector<uint32_t> occupied_indices;
    std::vector<int32_t>  occupied_positions;
    occupied_indices.reserve(kv_size);
    occupied_positions.reserve(kv_size);

    for (uint32_t i = 0; i < kv_size; i++) {
        if (state->cell_positions[i] >= 0) {
            occupied_indices.push_back(i);
            occupied_positions.push_back(state->cell_positions[i]);
        }
    }

    const uint32_t n_occupied = (uint32_t)occupied_indices.size();
    if (n_occupied <= budget) return 0;

    // ---- Step 2: Separate protected tokens from eviction candidates ----
    // Two protection classes:
    //   1. Prefix-protected: initial prompt tokens (if protect_prefill is set)
    //   2. Recent-protected: the most recent divide_length positions are never evicted.
    //      This ensures seq_pos_max remains unchanged after pruning, so the server's
    //      position counter (which expects Y = seq_pos_max + 1) stays consistent.
    //      Without this, evicting the highest-position token would cause
    //      "inconsistent sequence positions" errors.

    // Compute max position for recent-window protection
    int32_t max_pos = -1;
    for (uint32_t i = 0; i < n_occupied; i++) {
        if (occupied_positions[i] > max_pos) {
            max_pos = occupied_positions[i];
        }
    }

    const int32_t recent_threshold = max_pos - (int32_t)cfg.divide_length + 1;

    std::vector<uint32_t> decode_local_idx;   // index into occupied_indices
    std::vector<uint32_t> decode_cell_idx;    // actual cell indices
    std::vector<int32_t>  decode_positions;
    uint32_t n_protected = 0;  // prefix + recent protected count

    for (uint32_t i = 0; i < n_occupied; i++) {
        const bool is_prefix = cfg.protect_prefill &&
                               occupied_positions[i] < (int32_t)state->prefix_length;
        const bool is_recent = occupied_positions[i] >= recent_threshold;

        if (is_prefix || is_recent) {
            n_protected++;
        } else {
            decode_local_idx.push_back(i);
            decode_cell_idx.push_back(occupied_indices[i]);
            decode_positions.push_back(occupied_positions[i]);
        }
    }

    const uint32_t n_decode = (uint32_t)decode_cell_idx.size();
    const uint32_t decode_budget = (budget > n_protected) ? (budget - n_protected) : 0;

    if (n_decode <= decode_budget) return 0;

    // ---- Step 3: Score all sampled (layer, head) pairs ----
    // score_buf layout: [n_sampled, n_decode] — row-major
    float * score_buf = state->score_buf;

    // Determine K tensor type (used for lazy GPU init)
    const ggml_type k_type = (n_layers > 0 && k_tensors[0]) ? k_tensors[0]->type : GGML_TYPE_F32;

    // Lazy GPU init: runs only once per state lifetime
    if (!state->gpu_init_tried) {
        triattention_init_gpu(state, k_type);
    }

    if (state->use_gpu && !state->fallback_active) {
        // ---- GPU path ----
        // Upload the n_decode candidate cell indices + positions to device.
        // Kernels are enqueued into the default stream (nullptr), ordered after the upload.
        uint32_t * d_cell_indices = nullptr;
        int32_t  * d_positions    = nullptr;
        triattention_gpu_upload_cells(
            &d_cell_indices, &d_positions,
            decode_cell_idx.data(), decode_positions.data(),
            n_decode, nullptr);

        // Allocate a packed score buffer [n_sampled × n_decode] on device
        float * d_scores_all = triattention_gpu_alloc_scores(
            (uint32_t)((size_t)cal->n_sampled * n_decode), nullptr);

        // Launch one scoring kernel per sampled head (all async on default stream)
        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            const uint32_t layer_idx = cal->sampled_layer[sh];
            const uint32_t attn_head = cal->sampled_head[sh];
            const triattention_layer_params & layer = cal->layers[layer_idx];
            const uint32_t kv_source = layer.kv_source_layer;
            const uint32_t kv_head   = layer.num_kv_groups > 0 ? attn_head / layer.num_kv_groups : 0;

            int32_t ikv = (kv_source < cal->num_layers) ? layer_to_cache[kv_source] : -1;
            if (ikv < 0) {
                // Layer not in cache — will zero out after copy
                continue;
            }

            const ggml_tensor * kt = k_tensors[ikv];
            const size_t row_bytes = (size_t)(ggml_nbytes(kt) / (size_t)kt->ne[1]);
            const uint64_t n_embd  = (uint64_t)kt->ne[0];

            triattention_gpu_score_head(
                (triattention_gpu_state *)state->d_gpu_state,
                kt->data,                                   // device pointer (K on GPU)
                n_embd,
                row_bytes,
                kv_head,
                sh,                                         // head_calib_idx
                d_cell_indices,
                d_positions,
                n_decode,
                (int64_t)state->absolute_position,          // round_start
                (int)cfg.agg,
                d_scores_all + (size_t)sh * n_decode,       // output slice
                nullptr);                                    // default stream
        }

        // Copy all scores to host with one synchronization point
        triattention_gpu_scores_to_host(
            score_buf, d_scores_all,
            (uint32_t)((size_t)cal->n_sampled * n_decode), nullptr);

        // Zero out scores for heads whose layers were not in cache
        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            const uint32_t layer_idx = cal->sampled_layer[sh];
            const triattention_layer_params & layer = cal->layers[layer_idx];
            const uint32_t kv_source = layer.kv_source_layer;
            int32_t ikv = (kv_source < cal->num_layers) ? layer_to_cache[kv_source] : -1;
            if (ikv < 0) {
                memset(score_buf + (size_t)sh * n_decode, 0, n_decode * sizeof(float));
            }
        }

        triattention_gpu_free_dev(d_scores_all);
        triattention_gpu_free_dev(d_cell_indices);
        triattention_gpu_free_dev(d_positions);

    } else {
        // ---- CPU fallback path ----
        for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
            const uint32_t layer_idx = cal->sampled_layer[sh];
            const uint32_t attn_head = cal->sampled_head[sh];
            const triattention_layer_params & layer = cal->layers[layer_idx];
            const uint32_t kv_source = layer.kv_source_layer;
            const uint32_t kv_head   = layer.num_kv_groups > 0 ? attn_head / layer.num_kv_groups : 0;

            int32_t ikv = (kv_source < cal->num_layers) ? layer_to_cache[kv_source] : -1;
            if (ikv < 0) {
                // Layer not in cache (filtered out) — zero scores
                memset(score_buf + (size_t)sh * n_decode, 0, n_decode * sizeof(float));
                continue;
            }
            if (kv_head >= layer.num_kv_heads) {
                memset(score_buf + (size_t)sh * n_decode, 0, n_decode * sizeof(float));
                continue;
            }

            const ggml_tensor * k_tensor = k_tensors[ikv];
            const ggml_type k_type_l = k_tensor->type;
            const bool need_wht_inv = (k_type_l == GGML_TYPE_TURBO2_0 || k_type_l == GGML_TYPE_TURBO3_0);
            const uint32_t padded_hd = triattention_padded_dim(layer.head_dim);

            // 3a. Dequantize K for this KV head for all decode cells
            triattention_dequant_kv_head(
                state->dequant_buf,
                k_tensor,
                decode_cell_idx.data(),
                kv_head,
                n_decode,
                padded_hd,
                need_wht_inv);

            // 3b. Extract the rotary-tracked slice used by TriAttention.
            triattention_extract_rope_slice(
                state->rope_buf,
                state->dequant_buf,
                n_decode,
                padded_hd,
                layer.rope_offset,
                layer.rope_dim);

            // 3c. Invert RoPE → pre-RoPE K
            triattention_invert_rope(
                state->unrot_buf,
                state->rope_buf,
                decode_positions.data(),
                layer.omega,
                n_decode,
                layer.rope_dim,
                layer.freq_count,
                layer.rope_style);

            // 3d. Score keys
            if (state->fallback_active) {
                triattention_score_keys_norm(
                    score_buf + (size_t)sh * n_decode,
                    state->unrot_buf,
                    layer.freq_scale_sq,
                    n_decode,
                    layer.rope_dim,
                    layer.freq_count);
            } else {
                triattention_score_keys(
                    score_buf + (size_t)sh * n_decode,
                    state->unrot_buf,
                    &cal->head_stats[sh],
                    layer.omega,
                    layer.freq_scale_sq,
                    state->offsets,
                    decode_positions.data(),
                    state->absolute_position,
                    n_decode,
                    layer.rope_dim,
                    layer.freq_count,
                    state->n_offsets,
                    cfg.agg,
                    cfg.disable_trig);
            }
        }
    }

    std::vector<float> recency_scores;
    if (state->fallback_active) {
        recency_scores.resize(n_decode);
        triattention_build_recency_scores(recency_scores.data(), decode_positions.data(), n_decode);
        for (uint32_t sh = 0; sh < cal->n_sampled; ++sh) {
            float * row = score_buf + (size_t) sh * n_decode;
            triattention_blend_fallback_scores(
                row,
                recency_scores.data(),
                n_decode,
                state->cfg.fallback_recency_weight);
        }
    }

    // ---- Step 4: Combine scores across heads ----
    float * combined = state->combined_buf;

    if (cfg.mode == TRIATTENTION_MODE_GLOBAL) {
        // ---- Global union-based selection (Paper default) ----
        // 4a. Z-score normalize per head
        if (cfg.normalize_scores && !state->fallback_active) {
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                zscore_normalize(score_buf + (size_t)sh * n_decode, n_decode);
            }
        }

        // 4b. Add tie-breaking noise
        if (cfg.seed >= 0) {
            std::mt19937 rng((uint32_t)cfg.seed + (uint32_t)state->total_prune_calls);
            std::uniform_real_distribution<float> noise(-1e-6f, 1e-6f);
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                float * s = score_buf + (size_t)sh * n_decode;
                for (uint32_t i = 0; i < n_decode; i++) {
                    s[i] += noise(rng);
                }
            }
        }

        // 4c. Union selection:
        //   - Each head independently picks top-B
        //   - Union all selected indices
        //   - Combined score = max over all heads for each index
        //   - Keep top-B from union by combined score

        // First: compute max-over-heads score for each decode token
        for (uint32_t i = 0; i < n_decode; i++) {
            float max_score = -1e30f;
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                float s = score_buf[sh * n_decode + i];
                if (s > max_score) max_score = s;
            }
            combined[i] = max_score;
        }

        // 4d. Select top-B by combined score
        top_k_indices(state->keep_indices, combined, n_decode, decode_budget);

    } else if (cfg.mode == TRIATTENTION_MODE_PER_KV_HEAD) {
        // ---- Per-KV-head independent selection ----
        // Group sampled attention heads by KV head. For heterogeneous layouts
        // we fall back to per-(layer,kv_head) grouping because cross-layer KV
        // head identities are not comparable.
        std::vector<uint64_t> group_keys;
        std::vector<std::vector<uint32_t>> kv_head_to_sampled;
        for (uint32_t sh = 0; sh < cal->n_sampled; ++sh) {
            const uint32_t layer_idx = cal->sampled_layer[sh];
            const triattention_layer_params & layer = cal->layers[layer_idx];
            const uint32_t kv_h = layer.num_kv_groups > 0 ? cal->sampled_head[sh] / layer.num_kv_groups : 0;
            const uint64_t key = cal->heterogeneous_layout
                ? ((uint64_t) layer_idx << 32) | kv_h
                : (uint64_t) kv_h;

            auto it = std::find(group_keys.begin(), group_keys.end(), key);
            if (it == group_keys.end()) {
                group_keys.push_back(key);
                kv_head_to_sampled.push_back({ sh });
            } else {
                kv_head_to_sampled[(size_t) (it - group_keys.begin())].push_back(sh);
            }
        }

        // For each decode token: max score across all KV heads
        // (simplified: we use the per-KV-head max as the combined score)
        for (uint32_t i = 0; i < n_decode; i++) combined[i] = -1e30f;

        for (const auto & heads : kv_head_to_sampled) {
            if (heads.empty()) continue;

            for (uint32_t i = 0; i < n_decode; i++) {
                float max_val = -1e30f;
                for (uint32_t sh_idx : heads) {
                    float s = score_buf[sh_idx * n_decode + i];
                    if (cfg.normalize_scores) {
                        // Normalization already applied above? No, only for global mode.
                        // Apply inline for per-KV-head mode.
                    }
                    if (s > max_val) max_val = s;
                }
                if (max_val > combined[i]) combined[i] = max_val;
            }
        }

        // Normalize if requested
        if (cfg.normalize_scores && !state->fallback_active) {
            // Per-KV-head z-score normalization on the per-head scores
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                zscore_normalize(score_buf + (size_t)sh * n_decode, n_decode);
            }
            // Recompute combined after normalization
            for (uint32_t i = 0; i < n_decode; i++) combined[i] = -1e30f;
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                for (uint32_t i = 0; i < n_decode; i++) {
                    float s = score_buf[sh * n_decode + i];
                    if (s > combined[i]) combined[i] = s;
                }
            }
        }

        // Select top-B by combined score
        top_k_indices(state->keep_indices, combined, n_decode, decode_budget);

    } else if (cfg.mode == TRIATTENTION_MODE_PER_LAYER_HEAD) {
        // ---- Per-layer-per-head independent selection ----
        // Each (layer, KV head) selects independently.
        // Final combined score = mean of per-(layer,kv_head) scores

        if (cfg.normalize_scores && !state->fallback_active) {
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                zscore_normalize(score_buf + (size_t)sh * n_decode, n_decode);
            }
        }

        // Aggregate: mean score across all sampled heads for each token
        for (uint32_t i = 0; i < n_decode; i++) {
            float sum = 0.0f;
            for (uint32_t sh = 0; sh < cal->n_sampled; sh++) {
                sum += score_buf[sh * n_decode + i];
            }
            combined[i] = sum / (float)cal->n_sampled;
        }

        top_k_indices(state->keep_indices, combined, n_decode, decode_budget);
    }

    // ---- Step 5: Build keep set and evict ----
    // Convert keep_indices (which index into decode arrays) to actual cell indices
    std::vector<bool> keep_set(n_decode, false);
    for (uint32_t i = 0; i < decode_budget && i < n_decode; i++) {
        keep_set[state->keep_indices[i]] = true;
    }

    // Evict cells not in keep set
    uint32_t n_evicted = 0;
    for (uint32_t i = 0; i < n_decode; i++) {
        if (!keep_set[i]) {
            uint32_t cell_idx = decode_cell_idx[i];
            state->cell_positions[cell_idx] = -1;
            n_evicted++;
        }
    }

    // ---- Step 6: Update statistics ----
    double t_end = triattention_time_ms();
    state->total_prune_calls++;
    state->total_tokens_evicted += n_evicted;
    state->last_prune_time_ms = t_end - t_start;
    state->total_prune_time_ms += state->last_prune_time_ms;

    if (cfg.enable_logging) {
        fprintf(stderr, "[TriAttention] Pruned: %u → %u tokens (%u evicted, %u protected [prefix=%lld, recent=%d]), "
                "%.2f ms [%s], pos=%lld\n",
                n_occupied, n_occupied - n_evicted, n_evicted, n_protected,
                (long long)state->prefix_length, (int)cfg.divide_length,
                state->last_prune_time_ms, state->use_gpu ? "GPU" : "CPU",
                (long long)state->absolute_position);
    }

    return (int32_t)n_evicted;
}

// ============================================================================
// Monitoring
// ============================================================================

void triattention_print_stats(const triattention_state * state, FILE * stream) {
    if (!state) return;

    fprintf(stream, "\n=== TriAttention Statistics ===\n");
    fprintf(stream, "  Model:            %s\n", state->cal->model_name);
    fprintf(stream, "  Runtime mode:     %s\n", state->fallback_active ? "experimental fallback" : "calibrated");
    fprintf(stream, "  Budget:           %u tokens\n", state->cfg.budget);
    fprintf(stream, "  Pruning interval: %u tokens\n", state->cfg.divide_length);
    fprintf(stream, "  Pruning mode:     %s\n",
            state->cfg.mode == TRIATTENTION_MODE_GLOBAL         ? "global (union)" :
            state->cfg.mode == TRIATTENTION_MODE_PER_KV_HEAD    ? "per-KV-head" :
            state->cfg.mode == TRIATTENTION_MODE_PER_LAYER_HEAD ? "per-layer-per-head" : "unknown");
    fprintf(stream, "  Score aggregation: %s\n",
            state->cfg.agg == TRIATTENTION_AGG_MEAN ? "mean" : "max");
    fprintf(stream, "  Sampled heads:    %u of %u\n", state->cal->n_sampled, state->cal->num_attn_heads);
    if (state->fallback_active) {
        fprintf(stream, "  Fallback lambda:  %.3f\n", state->cfg.fallback_recency_weight);
    }
    fprintf(stream, "  Geometric offsets: %u (max %u)\n", state->n_offsets, state->cfg.offset_max);
    fprintf(stream, "  ---\n");
    fprintf(stream, "  Total prune calls:    %llu\n", (unsigned long long)state->total_prune_calls);
    fprintf(stream, "  Total tokens evicted: %llu\n", (unsigned long long)state->total_tokens_evicted);
    fprintf(stream, "  Total prune time:     %.2f ms\n", state->total_prune_time_ms);
    if (state->total_prune_calls > 0) {
        fprintf(stream, "  Avg time per prune:   %.2f ms\n",
                state->total_prune_time_ms / state->total_prune_calls);
        fprintf(stream, "  Avg tokens per prune: %.1f\n",
                (double)state->total_tokens_evicted / state->total_prune_calls);
    }
    fprintf(stream, "  Last prune time:      %.2f ms\n", state->last_prune_time_ms);
    fprintf(stream, "  Current position:     %lld\n", (long long)state->absolute_position);
    fprintf(stream, "===============================\n\n");
}
