#pragma once

#include "llama-triattention.h"

#include <cstddef>
#include <vector>

struct llama_model;
struct llama_cparams;

static constexpr const char * TRIATTENTION_GGUF_KEY = "triattention.calibration";

struct triattention_rope_params {
    uint32_t n_dims;
    uint32_t n_ctx_orig;

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    const float * freq_factors;
    uint32_t freq_factor_count;
};

bool triattention_build_rope_arrays(
    float * omega,
    float * freq_scale_sq,
    uint32_t freq_count,
    const triattention_rope_params * params);

void triattention_model_params_clear(triattention_model_params * model);
bool triattention_model_params_init(
    const llama_model * model,
    const llama_cparams * cparams,
    uint32_t kv_size,
    triattention_model_params * out);

triattention_calibration * triattention_calibration_load_from_buffer(
    const void * data,
    size_t size,
    bool verbose = true,
    const char * source_name = nullptr);

bool triattention_calibration_save_to_buffer(
    const triattention_calibration * cal,
    std::vector<uint8_t> & out);

triattention_calibration * triattention_calibration_load(const char * path, bool verbose = true);
bool triattention_calibration_save(const char * path, const triattention_calibration * cal);
void triattention_calibration_free(triattention_calibration * cal);

triattention_calibration * triattention_calibration_subset(
    const triattention_calibration * cal,
    const uint32_t * sampled_layers,
    uint32_t n_sampled_layers);

bool triattention_calibration_validate(
    const triattention_calibration * cal,
    const triattention_model_params * model,
    bool warn_rope_theta = true);

triattention_calibration * triattention_calibration_create_fallback(
    const triattention_model_params * model,
    const uint32_t * sampled_layers = nullptr,
    uint32_t n_sampled_layers = 0);
