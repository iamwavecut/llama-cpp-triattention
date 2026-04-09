#include "llama-triattention-calibration.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

static float triattention_tensor_get_f32(
    const ggml_tensor * t,
    const void * data,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
    const char * base = static_cast<const char *>(data) + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];

    switch (t->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(base);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(base));
        default:
            return 0.0f;
    }
}

bool triattention_is_query_rope_name(const char * name) {
    if (!name || name[0] == '\0') {
        return false;
    }

    return strncmp(name, "Qcur", 4) == 0 || strncmp(name, "q_pe", 4) == 0;
}

bool triattention_parse_layer_index(const char * name, uint32_t * layer_idx) {
    if (!name || !layer_idx) {
        return false;
    }

    const char * dash = strrchr(name, '-');
    if (!dash || dash[1] == '\0') {
        return false;
    }

    const char * p = dash + 1;
    if (!std::isdigit((unsigned char) *p)) {
        return false;
    }

    uint32_t value = 0;
    while (std::isdigit((unsigned char) *p)) {
        value = value * 10u + (uint32_t) (*p - '0');
        ++p;
    }

    *layer_idx = (uint32_t) value;
    return true;
}

bool triattention_extract_rope_params(
    const ggml_tensor * rope_tensor,
    triattention_rope_params * params,
    uint32_t * rope_style,
    std::string * error) {
    if (!rope_tensor || rope_tensor->op != GGML_OP_ROPE || !params || !rope_style) {
        if (error) {
            *error = "expected a GGML_OP_ROPE tensor";
        }
        return false;
    }

    const int32_t * op = reinterpret_cast<const int32_t *>(rope_tensor->op_params);
    const int mode = op[2];

    if (mode == GGML_ROPE_TYPE_NORMAL) {
        *rope_style = 1;
    } else if ((mode & GGML_ROPE_TYPE_NEOX) == GGML_ROPE_TYPE_NEOX) {
        *rope_style = 0;
    } else {
        if (error) {
            *error = "unsupported rope mode for calibration";
        }
        return false;
    }

    params->n_dims = (uint32_t) op[1];
    params->n_ctx_orig = (uint32_t) op[4];
    memcpy(&params->freq_base,   op + 5, sizeof(float));
    memcpy(&params->freq_scale,  op + 6, sizeof(float));
    memcpy(&params->ext_factor,  op + 7, sizeof(float));
    memcpy(&params->attn_factor, op + 8, sizeof(float));
    memcpy(&params->beta_fast,   op + 9, sizeof(float));
    memcpy(&params->beta_slow,   op + 10, sizeof(float));

    params->freq_factors = nullptr;
    params->freq_factor_count = 0;
    if (rope_tensor->src[2] != nullptr) {
        if (rope_tensor->src[2]->type != GGML_TYPE_F32) {
            if (error) {
                *error = "rope frequency factors must be F32";
            }
            return false;
        }
        params->freq_factors = static_cast<const float *>(rope_tensor->src[2]->data);
        params->freq_factor_count = (uint32_t) rope_tensor->src[2]->ne[0];
    }

    return true;
}

triattention_calibration_builder::triattention_calibration_builder(
    std::string model_name,
    uint32_t num_layers,
    uint32_t num_attn_heads,
    uint32_t num_kv_heads)
    : model_name_(std::move(model_name))
    , num_layers_(num_layers)
    , num_attn_heads_(num_attn_heads)
    , num_kv_heads_(num_kv_heads) {
}

void triattention_calibration_builder::note_rope_source(const char * name) {
    if (!name || name[0] == '\0') {
        return;
    }

    const std::string value(name);
    if (std::find(rope_sources_seen_.begin(), rope_sources_seen_.end(), value) == rope_sources_seen_.end()) {
        rope_sources_seen_.push_back(value);
    }
}

size_t triattention_calibration_builder::head_offset(uint32_t layer_idx, uint32_t head_idx, uint32_t freq_idx) const {
    return ((size_t) layer_idx * num_attn_heads_ + head_idx) * freq_count_ + freq_idx;
}

bool triattention_calibration_builder::ensure_layout(
    const ggml_tensor * src0,
    uint32_t layer_idx,
    uint32_t rope_style,
    std::string * error) {
    if (!src0) {
        if (error) {
            *error = "missing rope source tensor";
        }
        return false;
    }

    if (src0->ne[3] != 1) {
        if (error) {
            *error = "only single-batch decoder-only query tensors are supported";
        }
        return false;
    }
    if (src0->ne[0] <= 0 || (src0->ne[0] % 2) != 0) {
        if (error) {
            *error = "query tensor head_dim must be even";
        }
        return false;
    }
    if ((uint32_t) src0->ne[1] != num_attn_heads_) {
        if (error) {
            *error = "query tensor attention head count does not match model metadata";
        }
        return false;
    }
    if (layer_idx >= num_layers_) {
        if (error) {
            *error = "layer index out of range";
        }
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        if (error) {
            *error = "query tensor type must be F32/F16/BF16";
        }
        return false;
    }

    const uint32_t observed_head_dim = (uint32_t) src0->ne[0];
    if (head_dim_ == 0) {
        head_dim_ = observed_head_dim;
        freq_count_ = head_dim_ / 2;
        rope_style_ = rope_style;

        const size_t total = (size_t) num_layers_ * num_attn_heads_ * freq_count_;
        sum_real_.assign(total, 0.0);
        sum_imag_.assign(total, 0.0);
        sum_abs_.assign(total, 0.0);
        counts_.assign((size_t) num_layers_ * num_attn_heads_, 0);
    } else if (head_dim_ != observed_head_dim || rope_style_ != rope_style) {
        if (error) {
            *error = "inconsistent query tensor layout across layers";
        }
        return false;
    }

    return true;
}

bool triattention_calibration_builder::ensure_rope_arrays(
    const triattention_rope_params & rope_params,
    std::string * error) {
    std::vector<float> omega(freq_count_);
    std::vector<float> freq_scale_sq(freq_count_);
    if (!triattention_build_rope_arrays(omega.data(), freq_scale_sq.data(), freq_count_, &rope_params)) {
        if (error) {
            *error = "failed to derive RoPE frequencies";
        }
        return false;
    }

    if (omega_.empty()) {
        omega_ = std::move(omega);
        freq_scale_sq_ = std::move(freq_scale_sq);
        rope_theta_ = rope_params.freq_base > 0.0f ? rope_params.freq_base : 10000.0f;
        return true;
    }

    for (uint32_t i = 0; i < freq_count_; ++i) {
        if (fabsf(omega_[i] - omega[i]) > 1e-6f || fabsf(freq_scale_sq_[i] - freq_scale_sq[i]) > 1e-6f) {
            if (error) {
                *error = "per-layer rotary frequency variation is not supported by this calibrator";
            }
            return false;
        }
    }

    return true;
}

bool triattention_calibration_builder::accumulate_query_tensor(
    const ggml_tensor * src0,
    const void * src0_data,
    uint32_t layer_idx,
    uint32_t rope_style,
    const triattention_rope_params & rope_params,
    std::string * error) {
    if (!ensure_layout(src0, layer_idx, rope_style, error)) {
        return false;
    }
    if (!ensure_rope_arrays(rope_params, error)) {
        return false;
    }

    const uint32_t n_tokens = (uint32_t) src0->ne[2];
    for (uint32_t head = 0; head < num_attn_heads_; ++head) {
        for (uint32_t tok = 0; tok < n_tokens; ++tok) {
            for (uint32_t f = 0; f < freq_count_; ++f) {
                const float re = rope_style_ == 0
                    ? triattention_tensor_get_f32(src0, src0_data, f, head, tok, 0)
                    : triattention_tensor_get_f32(src0, src0_data, 2 * f, head, tok, 0);
                const float im = rope_style_ == 0
                    ? triattention_tensor_get_f32(src0, src0_data, f + freq_count_, head, tok, 0)
                    : triattention_tensor_get_f32(src0, src0_data, 2 * f + 1, head, tok, 0);

                const size_t off = head_offset(layer_idx, head, f);
                sum_real_[off] += re;
                sum_imag_[off] += im;
                sum_abs_[off] += sqrtf(re * re + im * im);
            }

            counts_[(size_t) layer_idx * num_attn_heads_ + head] += 1;
        }
    }

    return true;
}

triattention_calibration * triattention_calibration_builder::finalize(std::string * error) const {
    if (!has_captured_queries()) {
        if (error) {
            *error = "no query rope tensors were captured";
        }
        return nullptr;
    }

    auto * cal = new triattention_calibration();
    memset(cal, 0, sizeof(*cal));

    cal->version        = TRIATTENTION_VERSION;
    cal->head_dim       = head_dim_;
    cal->num_layers     = num_layers_;
    cal->num_attn_heads = num_attn_heads_;
    cal->num_kv_heads   = num_kv_heads_;
    cal->num_kv_groups  = num_attn_heads_ / num_kv_heads_;
    cal->rope_theta     = rope_theta_;
    cal->rope_style     = rope_style_;
    cal->freq_count     = freq_count_;

    uint32_t n_sampled = 0;
    for (uint32_t layer = 0; layer < num_layers_; ++layer) {
        for (uint32_t head = 0; head < num_attn_heads_; ++head) {
            if (counts_[(size_t) layer * num_attn_heads_ + head] > 0) {
                ++n_sampled;
            }
        }
    }
    cal->n_sampled = n_sampled;

    snprintf(cal->model_name, sizeof(cal->model_name), "%s", model_name_.c_str());

    cal->omega = new float[freq_count_];
    cal->freq_scale_sq = new float[freq_count_];
    memcpy(cal->omega, omega_.data(), sizeof(float) * freq_count_);
    memcpy(cal->freq_scale_sq, freq_scale_sq_.data(), sizeof(float) * freq_count_);

    cal->sampled_layer = new uint32_t[n_sampled];
    cal->sampled_head  = new uint32_t[n_sampled];
    cal->head_stats    = new triattention_head_stats[n_sampled];
    memset(cal->head_stats, 0, sizeof(triattention_head_stats) * n_sampled);

    uint32_t out_idx = 0;
    for (uint32_t layer = 0; layer < num_layers_; ++layer) {
        for (uint32_t head = 0; head < num_attn_heads_; ++head) {
            const uint64_t count = counts_[(size_t) layer * num_attn_heads_ + head];
            if (count == 0) {
                continue;
            }

            cal->sampled_layer[out_idx] = layer;
            cal->sampled_head[out_idx]  = head;

            auto & hs = cal->head_stats[out_idx];
            hs.q_mean_real = new float[freq_count_];
            hs.q_mean_imag = new float[freq_count_];
            hs.q_abs_mean  = new float[freq_count_];
            hs.r_f         = new float[freq_count_];

            const double inv_count = 1.0 / (double) count;
            for (uint32_t f = 0; f < freq_count_; ++f) {
                const size_t off = head_offset(layer, head, f);
                const float mean_real = (float) (sum_real_[off] * inv_count);
                const float mean_imag = (float) (sum_imag_[off] * inv_count);
                const float abs_mean = (float) (sum_abs_[off] * inv_count);
                const float mean_abs = sqrtf(mean_real * mean_real + mean_imag * mean_imag);

                hs.q_mean_real[f] = mean_real;
                hs.q_mean_imag[f] = mean_imag;
                hs.q_abs_mean[f]  = abs_mean;
                hs.r_f[f]         = abs_mean > 0.0f ? mean_abs / abs_mean : 0.0f;
            }

            ++out_idx;
        }
    }

    return cal;
}

bool triattention_calibration_builder::has_captured_queries() const {
    return !counts_.empty() && std::any_of(counts_.begin(), counts_.end(), [](uint64_t count) {
        return count > 0;
    });
}

const std::vector<std::string> & triattention_calibration_builder::rope_sources_seen() const {
    return rope_sources_seen_;
}
