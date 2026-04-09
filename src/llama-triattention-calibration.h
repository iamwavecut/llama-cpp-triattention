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
        uint32_t num_layers,
        uint32_t num_attn_heads,
        uint32_t num_kv_heads);

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
    size_t head_offset(uint32_t layer_idx, uint32_t head_idx, uint32_t freq_idx) const;
    bool ensure_layout(const ggml_tensor * src0, uint32_t layer_idx, uint32_t rope_style, std::string * error);
    bool ensure_rope_arrays(const triattention_rope_params & rope_params, std::string * error);

    std::string model_name_;
    uint32_t num_layers_;
    uint32_t num_attn_heads_;
    uint32_t num_kv_heads_;

    uint32_t head_dim_ = 0;
    uint32_t freq_count_ = 0;
    uint32_t rope_style_ = 0;
    double rope_theta_ = 10000.0;

    std::vector<double> sum_real_;
    std::vector<double> sum_imag_;
    std::vector<double> sum_abs_;
    std::vector<uint64_t> counts_;
    std::vector<float> omega_;
    std::vector<float> freq_scale_sq_;
    std::vector<std::string> rope_sources_seen_;
};
