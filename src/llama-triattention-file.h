#pragma once

#include "llama-triattention.h"

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

triattention_calibration * triattention_calibration_load(const char * path, bool verbose = true);
bool triattention_calibration_save(const char * path, const triattention_calibration * cal);
void triattention_calibration_free(triattention_calibration * cal);

bool triattention_calibration_validate(
    const triattention_calibration * cal,
    const triattention_model_params * model,
    bool warn_rope_theta = true);

triattention_calibration * triattention_calibration_create_fallback(
    const triattention_model_params * model);
