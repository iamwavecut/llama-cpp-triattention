#include "llama-triattention-calibration.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>

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

    *layer_idx = value;
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
    const triattention_model_params * model)
    : model_name_(std::move(model_name))
    , model_(model) {
    if (model_) {
        layers_.resize(model_->num_layers);
    }
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

size_t triattention_calibration_builder::head_offset(const layer_accum & layer, uint32_t head_idx, uint32_t freq_idx) const {
    return (size_t) head_idx * layer.freq_count + freq_idx;
}

bool triattention_calibration_builder::ensure_layout(
    const ggml_tensor * src0,
    uint32_t layer_idx,
    uint32_t rope_style,
    const triattention_rope_params & rope_params,
    std::string * error) {
    if (!model_ || !model_->layers || layer_idx >= model_->num_layers) {
        if (error) {
            *error = "layer index out of range";
        }
        return false;
    }
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
            *error = "query tensor rope_dim must be even";
        }
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        if (error) {
            *error = "query tensor type must be F32/F16/BF16";
        }
        return false;
    }

    const triattention_layer_params & expected = model_->layers[layer_idx];
    if ((uint32_t) src0->ne[1] != expected.num_attn_heads) {
        if (error) {
            *error = "query tensor attention head count does not match model metadata";
        }
        return false;
    }
    if ((uint32_t) src0->ne[0] != expected.rope_dim) {
        if (error) {
            *error = "query tensor rope_dim does not match model metadata";
        }
        return false;
    }
    if ((uint32_t) rope_params.n_dims != expected.rope_dim) {
        if (error) {
            *error = "rope op n_dims does not match model metadata";
        }
        return false;
    }
    if (rope_style != expected.rope_style) {
        if (error) {
            *error = "rope style does not match model metadata";
        }
        return false;
    }

    layer_accum & layer = layers_[layer_idx];
    if (layer.num_attn_heads == 0) {
        layer.head_dim         = expected.head_dim;
        layer.rope_dim         = expected.rope_dim;
        layer.rope_offset      = expected.rope_offset;
        layer.num_attn_heads   = expected.num_attn_heads;
        layer.num_kv_heads     = expected.num_kv_heads;
        layer.num_kv_groups    = expected.num_kv_groups;
        layer.kv_source_layer  = expected.kv_source_layer;
        layer.rope_style       = expected.rope_style;
        layer.freq_count       = expected.freq_count;
        layer.n_ctx_orig       = expected.n_ctx_orig;
        layer.rope_theta       = expected.rope_theta;
        layer.rope_freq_scale  = expected.rope_freq_scale;
        layer.rope_ext_factor  = expected.rope_ext_factor;
        layer.rope_attn_factor = expected.rope_attn_factor;
        layer.rope_beta_fast   = expected.rope_beta_fast;
        layer.rope_beta_slow   = expected.rope_beta_slow;
        layer.sum_real.assign((size_t) layer.num_attn_heads * layer.freq_count, 0.0);
        layer.sum_imag.assign((size_t) layer.num_attn_heads * layer.freq_count, 0.0);
        layer.sum_abs.assign((size_t) layer.num_attn_heads * layer.freq_count, 0.0);
        layer.counts.assign(layer.num_attn_heads, 0);
        layer.omega.resize(layer.freq_count);
        layer.freq_scale_sq.resize(layer.freq_count);
        if (!triattention_build_rope_arrays(layer.omega.data(), layer.freq_scale_sq.data(), layer.freq_count, &rope_params)) {
            if (error) {
                *error = "failed to derive RoPE frequencies";
            }
            return false;
        }
    } else {
        if (layer.rope_style != rope_style || layer.rope_dim != (uint32_t) src0->ne[0] ||
            layer.num_attn_heads != (uint32_t) src0->ne[1]) {
            if (error) {
                *error = "inconsistent query tensor layout across captures";
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
    if (!ensure_layout(src0, layer_idx, rope_style, rope_params, error)) {
        return false;
    }

    layer_accum & layer = layers_[layer_idx];
    const uint32_t n_tokens = (uint32_t) src0->ne[2];
    for (uint32_t head = 0; head < layer.num_attn_heads; ++head) {
        for (uint32_t tok = 0; tok < n_tokens; ++tok) {
            for (uint32_t f = 0; f < layer.freq_count; ++f) {
                const float re = layer.rope_style == 0
                    ? triattention_tensor_get_f32(src0, src0_data, f, head, tok, 0)
                    : triattention_tensor_get_f32(src0, src0_data, 2 * f, head, tok, 0);
                const float im = layer.rope_style == 0
                    ? triattention_tensor_get_f32(src0, src0_data, f + layer.freq_count, head, tok, 0)
                    : triattention_tensor_get_f32(src0, src0_data, 2 * f + 1, head, tok, 0);

                const size_t off = head_offset(layer, head, f);
                layer.sum_real[off] += re;
                layer.sum_imag[off] += im;
                layer.sum_abs[off] += sqrtf(re * re + im * im);
            }

            layer.counts[head] += 1;
        }
    }

    layer.captured = true;
    return true;
}

static bool triattention_copy_layer_params_local(
    triattention_layer_params * dst,
    const triattention_layer_params * src) {
    *dst = *src;
    dst->omega = nullptr;
    dst->freq_scale_sq = nullptr;

    if (src->freq_count > 0 && src->omega) {
        dst->omega = new float[src->freq_count];
        memcpy(dst->omega, src->omega, sizeof(float) * src->freq_count);
    }
    if (src->freq_count > 0 && src->freq_scale_sq) {
        dst->freq_scale_sq = new float[src->freq_count];
        memcpy(dst->freq_scale_sq, src->freq_scale_sq, sizeof(float) * src->freq_count);
    }
    return true;
}

triattention_calibration * triattention_calibration_builder::finalize(std::string * error) const {
    if (!model_ || !has_captured_queries()) {
        if (error) {
            *error = "no query rope tensors were captured";
        }
        return nullptr;
    }

    auto * cal = new triattention_calibration();
    memset(cal, 0, sizeof(*cal));

    cal->version    = TRIATTENTION_VERSION;
    cal->num_layers = model_->num_layers;
    cal->layers     = new triattention_layer_params[cal->num_layers];
    memset(cal->layers, 0, sizeof(triattention_layer_params) * cal->num_layers);

    uint32_t max_head_dim = 0;
    uint32_t max_rope_dim = 0;
    uint32_t max_freq_count = 0;
    uint32_t max_attn_heads = 0;
    uint32_t max_kv_heads = 0;

    for (uint32_t il = 0; il < model_->num_layers; ++il) {
        triattention_layer_params layer = model_->layers[il];
        layer.omega = nullptr;
        layer.freq_scale_sq = nullptr;

        const layer_accum & acc = layers_[il];
        if (acc.captured && !acc.omega.empty() && !acc.freq_scale_sq.empty()) {
            layer.omega = new float[acc.freq_count];
            layer.freq_scale_sq = new float[acc.freq_count];
            memcpy(layer.omega, acc.omega.data(), sizeof(float) * acc.freq_count);
            memcpy(layer.freq_scale_sq, acc.freq_scale_sq.data(), sizeof(float) * acc.freq_count);
        } else if (model_->layers[il].omega && model_->layers[il].freq_scale_sq) {
            layer.omega = new float[layer.freq_count];
            layer.freq_scale_sq = new float[layer.freq_count];
            memcpy(layer.omega, model_->layers[il].omega, sizeof(float) * layer.freq_count);
            memcpy(layer.freq_scale_sq, model_->layers[il].freq_scale_sq, sizeof(float) * layer.freq_count);
        }

        triattention_copy_layer_params_local(&cal->layers[il], &layer);
        delete[] layer.omega;
        delete[] layer.freq_scale_sq;

        max_head_dim   = std::max(max_head_dim, cal->layers[il].head_dim);
        max_rope_dim   = std::max(max_rope_dim, cal->layers[il].rope_dim);
        max_freq_count = std::max(max_freq_count, cal->layers[il].freq_count);
        max_attn_heads = std::max(max_attn_heads, cal->layers[il].num_attn_heads);
        max_kv_heads   = std::max(max_kv_heads, cal->layers[il].num_kv_heads);
    }

    cal->max_head_dim   = max_head_dim;
    cal->max_rope_dim   = max_rope_dim;
    cal->max_freq_count = max_freq_count;
    cal->head_dim       = max_head_dim;
    cal->freq_count     = max_freq_count;
    cal->num_attn_heads = max_attn_heads;
    cal->num_kv_heads   = max_kv_heads;
    cal->rope_theta     = cal->num_layers > 0 ? cal->layers[0].rope_theta : 10000.0;
    cal->rope_style     = cal->num_layers > 0 ? cal->layers[0].rope_style : 0;
    cal->num_kv_groups  = cal->num_layers > 0 ? cal->layers[0].num_kv_groups : 0;
    cal->heterogeneous_layout = model_->heterogeneous_layout;

    uint32_t n_sampled = 0;
    for (uint32_t layer = 0; layer < model_->num_layers; ++layer) {
        const layer_accum & acc = layers_[layer];
        for (uint32_t head = 0; head < acc.counts.size(); ++head) {
            if (acc.counts[head] > 0) {
                ++n_sampled;
            }
        }
    }
    cal->n_sampled = n_sampled;

    snprintf(cal->model_name, sizeof(cal->model_name), "%s", model_name_.c_str());

    if (!cal->heterogeneous_layout && cal->num_layers > 0 && cal->layers[0].omega && cal->layers[0].freq_scale_sq) {
        cal->omega = new float[cal->layers[0].freq_count];
        cal->freq_scale_sq = new float[cal->layers[0].freq_count];
        memcpy(cal->omega, cal->layers[0].omega, sizeof(float) * cal->layers[0].freq_count);
        memcpy(cal->freq_scale_sq, cal->layers[0].freq_scale_sq, sizeof(float) * cal->layers[0].freq_count);
    }

    cal->sampled_layer = new uint32_t[n_sampled];
    cal->sampled_head  = new uint32_t[n_sampled];
    cal->head_stats    = new triattention_head_stats[n_sampled];
    memset(cal->head_stats, 0, sizeof(triattention_head_stats) * n_sampled);

    uint32_t out_idx = 0;
    for (uint32_t layer = 0; layer < model_->num_layers; ++layer) {
        const layer_accum & acc = layers_[layer];
        if (!acc.captured) {
            continue;
        }

        for (uint32_t head = 0; head < acc.num_attn_heads; ++head) {
            const uint64_t count = acc.counts[head];
            if (count == 0) {
                continue;
            }

            cal->sampled_layer[out_idx] = layer;
            cal->sampled_head[out_idx]  = head;

            triattention_head_stats & hs = cal->head_stats[out_idx];
            hs.q_mean_real = new float[acc.freq_count];
            hs.q_mean_imag = new float[acc.freq_count];
            hs.q_abs_mean  = new float[acc.freq_count];
            hs.r_f         = new float[acc.freq_count];

            const double inv_count = 1.0 / (double) count;
            for (uint32_t f = 0; f < acc.freq_count; ++f) {
                const size_t off = head_offset(acc, head, f);
                const float mean_real = (float) (acc.sum_real[off] * inv_count);
                const float mean_imag = (float) (acc.sum_imag[off] * inv_count);
                const float abs_mean = (float) (acc.sum_abs[off] * inv_count);
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
    return std::any_of(layers_.begin(), layers_.end(), [](const layer_accum & layer) {
        return layer.captured && std::any_of(layer.counts.begin(), layer.counts.end(), [](uint64_t count) {
            return count > 0;
        });
    });
}

const std::vector<std::string> & triattention_calibration_builder::rope_sources_seen() const {
    return rope_sources_seen_;
}
