#pragma once

#include "ggml.h"
#include "llama-triattention-file.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

bool triattention_is_query_rope_name(const char * name);
bool triattention_parse_layer_index(const char * name, uint32_t * layer_idx);
bool triattention_extract_rope_params(
    const ggml_tensor * rope_tensor,
    triattention_rope_params * params,
    uint32_t * rope_style,
    std::string * error);

class triattention_calibration_builder {
public:
    triattention_calibration_builder(
        std::string model_name,
        const triattention_model_params * model);

    void note_rope_source(const char * name);

    bool accumulate_query_tensor(
        const ggml_tensor * src0,
        const void * src0_data,
        uint32_t layer_idx,
        uint32_t rope_style,
        const triattention_rope_params & rope_params,
        std::string * error);

    triattention_calibration * finalize(std::string * error) const;

    bool has_captured_queries() const;
    const std::vector<std::string> & rope_sources_seen() const;

private:
    struct layer_accum {
        uint32_t head_dim = 0;
        uint32_t rope_dim = 0;
        uint32_t rope_offset = 0;
        uint32_t num_attn_heads = 0;
        uint32_t num_kv_heads = 0;
        uint32_t num_kv_groups = 0;
        uint32_t kv_source_layer = 0;
        uint32_t rope_style = 0;
        uint32_t freq_count = 0;
        uint32_t n_ctx_orig = 0;
        double rope_theta = 10000.0;
        float rope_freq_scale = 1.0f;
        float rope_ext_factor = 0.0f;
        float rope_attn_factor = 1.0f;
        float rope_beta_fast = 32.0f;
        float rope_beta_slow = 1.0f;
        bool captured = false;

        std::vector<double> sum_real;
        std::vector<double> sum_imag;
        std::vector<double> sum_abs;
        std::vector<uint64_t> counts;
        std::vector<float> omega;
        std::vector<float> freq_scale_sq;
    };

    size_t head_offset(const layer_accum & layer, uint32_t head_idx, uint32_t freq_idx) const;
    bool ensure_layout(const ggml_tensor * src0, uint32_t layer_idx, uint32_t rope_style, const triattention_rope_params & rope_params, std::string * error);

    std::string model_name_;
    const triattention_model_params * model_ = nullptr;
    std::vector<layer_accum> layers_;
    std::vector<std::string> rope_sources_seen_;
};
