#include "llama-triattention-file.h"

#include "llama-cparams.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static constexpr uint32_t TRIATTENTION_VERSION_V1 = 1u;
static constexpr uint32_t TRIATTENTION_VERSION_V2 = 2u;
static constexpr uint32_t TRIATTENTION_LAYER_NONE = std::numeric_limits<uint32_t>::max();

struct triattention_binary_reader {
    const uint8_t * data = nullptr;
    size_t size = 0;
    size_t offset = 0;
    const char * source = "<memory>";

    bool read(void * dst, size_t nbytes) {
        if (!dst || offset > size || nbytes > size - offset) {
            return false;
        }
        memcpy(dst, data + offset, nbytes);
        offset += nbytes;
        return true;
    }
};

struct triattention_binary_writer {
    std::vector<uint8_t> * out = nullptr;

    bool write(const void * src, size_t nbytes) {
        if (!out) {
            return false;
        }
        const size_t base = out->size();
        out->resize(base + nbytes);
        memcpy(out->data() + base, src, nbytes);
        return true;
    }
};

static float triattention_rope_yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2.0f - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static float triattention_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    static const float kPi = 3.14159265358979323846f;
    return n_dims * logf((float) n_ctx_orig / (n_rot * 2.0f * kPi)) / (2.0f * logf(base));
}

static void triattention_rope_yarn_corr_dims(
    int n_dims,
    int n_ctx_orig,
    float freq_base,
    float beta_fast,
    float beta_slow,
    float corr_dims[2]) {
    const float start = floorf(triattention_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end   = ceilf(triattention_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    corr_dims[0] = start;
    corr_dims[1] = end;
}

static void triattention_rope_yarn(
    float theta_extrap,
    float freq_scale,
    float corr_dims[2],
    int i0,
    float ext_factor,
    float mscale,
    float * cos_theta,
    float * sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = triattention_rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void triattention_clear_layer_params(triattention_layer_params * layer) {
    if (!layer) {
        return;
    }

    delete[] layer->omega;
    delete[] layer->freq_scale_sq;
    layer->omega = nullptr;
    layer->freq_scale_sq = nullptr;
}

static void triattention_clear_layer_params_array(triattention_layer_params * layers, uint32_t num_layers) {
    if (!layers) {
        return;
    }

    for (uint32_t il = 0; il < num_layers; ++il) {
        triattention_clear_layer_params(&layers[il]);
    }
    delete[] layers;
}

static bool triattention_copy_layer_params(
    triattention_layer_params * dst,
    const triattention_layer_params * src) {
    if (!dst || !src) {
        return false;
    }

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

static bool triattention_layer_params_equal(const triattention_layer_params & a, const triattention_layer_params & b) {
    if (a.head_dim       != b.head_dim ||
        a.rope_dim       != b.rope_dim ||
        a.rope_offset    != b.rope_offset ||
        a.num_attn_heads != b.num_attn_heads ||
        a.num_kv_heads   != b.num_kv_heads ||
        a.num_kv_groups  != b.num_kv_groups ||
        a.rope_style     != b.rope_style ||
        a.freq_count     != b.freq_count) {
        return false;
    }

    if (fabs(a.rope_theta - b.rope_theta) > 1e-6) {
        return false;
    }

    if (fabsf(a.rope_freq_scale  - b.rope_freq_scale)  > 1e-6f ||
        fabsf(a.rope_ext_factor  - b.rope_ext_factor)  > 1e-6f ||
        fabsf(a.rope_attn_factor - b.rope_attn_factor) > 1e-6f ||
        fabsf(a.rope_beta_fast   - b.rope_beta_fast)   > 1e-6f ||
        fabsf(a.rope_beta_slow   - b.rope_beta_slow)   > 1e-6f) {
        return false;
    }

    if ((a.omega == nullptr) != (b.omega == nullptr) ||
        (a.freq_scale_sq == nullptr) != (b.freq_scale_sq == nullptr)) {
        return false;
    }

    for (uint32_t f = 0; a.omega && f < a.freq_count; ++f) {
        if (fabsf(a.omega[f] - b.omega[f]) > 1e-6f) {
            return false;
        }
    }

    for (uint32_t f = 0; a.freq_scale_sq && f < a.freq_count; ++f) {
        if (fabsf(a.freq_scale_sq[f] - b.freq_scale_sq[f]) > 1e-6f) {
            return false;
        }
    }

    return true;
}

static void triattention_update_summary_from_layers(
    triattention_calibration * cal,
    const triattention_layer_params * layers,
    uint32_t num_layers) {
    cal->head_dim       = 0;
    cal->num_attn_heads = 0;
    cal->num_kv_heads   = 0;
    cal->freq_count     = 0;
    cal->max_head_dim   = 0;
    cal->max_rope_dim   = 0;
    cal->max_freq_count = 0;
    cal->heterogeneous_layout = false;

    if (!layers || num_layers == 0) {
        cal->num_kv_groups = 0;
        cal->rope_theta = 10000.0;
        cal->rope_style = 0;
        return;
    }

    const triattention_layer_params & base = layers[0];
    cal->rope_theta = base.rope_theta;
    cal->rope_style = base.rope_style;

    for (uint32_t il = 0; il < num_layers; ++il) {
        const triattention_layer_params & layer = layers[il];
        cal->max_head_dim   = std::max(cal->max_head_dim, layer.head_dim);
        cal->max_rope_dim   = std::max(cal->max_rope_dim, layer.rope_dim);
        cal->max_freq_count = std::max(cal->max_freq_count, layer.freq_count);
        cal->head_dim       = std::max(cal->head_dim, layer.head_dim);
        cal->num_attn_heads = std::max(cal->num_attn_heads, layer.num_attn_heads);
        cal->num_kv_heads   = std::max(cal->num_kv_heads, layer.num_kv_heads);
        cal->freq_count     = std::max(cal->freq_count, layer.freq_count);
        if (il > 0 && !triattention_layer_params_equal(base, layer)) {
            cal->heterogeneous_layout = true;
        }
    }

    cal->num_kv_groups = cal->heterogeneous_layout ? 0 : base.num_kv_groups;
}

static void triattention_free_head_stats(triattention_head_stats * hs) {
    if (!hs) {
        return;
    }

    delete[] hs->q_mean_real;
    delete[] hs->q_mean_imag;
    delete[] hs->q_abs_mean;
    delete[] hs->r_f;
    delete[] hs->q_mean_abs;
    delete[] hs->extra_weight;

    hs->q_mean_real  = nullptr;
    hs->q_mean_imag  = nullptr;
    hs->q_abs_mean   = nullptr;
    hs->r_f          = nullptr;
    hs->q_mean_abs   = nullptr;
    hs->extra_weight = nullptr;
}

static uint32_t triattention_map_rope_style(enum llama_rope_type rope_type) {
    switch (rope_type) {
        case LLAMA_ROPE_TYPE_NORM:
            return 1;
        case LLAMA_ROPE_TYPE_NEOX:
        case LLAMA_ROPE_TYPE_MROPE:
        case LLAMA_ROPE_TYPE_IMROPE:
            return 0;
        default:
            return TRIATTENTION_LAYER_NONE;
    }
}

static uint32_t triattention_resolve_kv_source_layer(const llama_model & model, uint32_t il) {
    const llama_hparams & hparams = model.hparams;

    if (hparams.has_kv(il)) {
        return il;
    }

    if ((model.arch == LLM_ARCH_GEMMA3N || model.arch == LLM_ARCH_GEMMA4) &&
        hparams.n_layer_kv_from_start >= 0 &&
        il >= (uint32_t) hparams.n_layer_kv_from_start) {
        const int32_t reuse = (int32_t) hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
        if (reuse >= 0) {
            return (uint32_t) reuse;
        }
    }

    return TRIATTENTION_LAYER_NONE;
}

bool triattention_build_rope_arrays(
    float * omega,
    float * freq_scale_sq,
    uint32_t freq_count,
    const triattention_rope_params * params) {
    if (!omega || !freq_scale_sq || !params || params->n_dims == 0 || (params->n_dims % 2) != 0) {
        return false;
    }

    const float freq_base   = params->freq_base > 0.0f ? params->freq_base : 10000.0f;
    const float freq_scale  = params->freq_scale > 0.0f ? params->freq_scale : 1.0f;
    const float ext_factor  = params->ext_factor > 0.0f ? params->ext_factor : 0.0f;
    const float attn_factor = params->attn_factor > 0.0f ? params->attn_factor : 1.0f;
    const float beta_fast   = params->beta_fast > 0.0f ? params->beta_fast : 32.0f;
    const float beta_slow   = params->beta_slow > 0.0f ? params->beta_slow : 1.0f;
    const float theta_scale = powf(freq_base, -2.0f / params->n_dims);

    float corr_dims[2] = { 0.0f, 0.0f };
    const uint32_t n_ctx_orig = params->n_ctx_orig > 0 ? params->n_ctx_orig : 1;
    triattention_rope_yarn_corr_dims((int) params->n_dims, (int) n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    float theta = 1.0f;
    for (uint32_t f = 0; f < freq_count; ++f) {
        const float ff = params->freq_factors && f < params->freq_factor_count && params->freq_factors[f] != 0.0f
            ? params->freq_factors[f]
            : 1.0f;

        float cos1 = 0.0f;
        float sin1 = 0.0f;
        triattention_rope_yarn(theta / ff, freq_scale, corr_dims, (int) (f * 2), ext_factor, attn_factor, &cos1, &sin1);
        omega[f] = atan2f(sin1, cos1);

        float cos0 = 0.0f;
        float sin0 = 0.0f;
        triattention_rope_yarn(0.0f, freq_scale, corr_dims, (int) (f * 2), ext_factor, attn_factor, &cos0, &sin0);
        freq_scale_sq[f] = cos0 * cos0 + sin0 * sin0;

        theta *= theta_scale;
    }

    return true;
}

void triattention_model_params_clear(triattention_model_params * model) {
    if (!model) {
        return;
    }

    triattention_clear_layer_params_array(model->layers, model->num_layers);
    memset(model, 0, sizeof(*model));
}

bool triattention_model_params_init(
    const llama_model * model,
    const llama_cparams * cparams,
    uint32_t kv_size,
    triattention_model_params * out) {
    if (!model || !cparams || !out) {
        return false;
    }

    triattention_model_params_clear(out);

    const llama_hparams & hparams = model->hparams;
    const uint32_t rope_style = triattention_map_rope_style(hparams.rope_type);
    if (rope_style == TRIATTENTION_LAYER_NONE) {
        return false;
    }

    out->kv_size = kv_size;
    out->num_layers = hparams.n_layer;
    out->layers = new triattention_layer_params[out->num_layers];
    memset(out->layers, 0, sizeof(triattention_layer_params) * out->num_layers);

    for (uint32_t il = 0; il < out->num_layers; ++il) {
        triattention_layer_params & layer = out->layers[il];

        const uint32_t kv_source = triattention_resolve_kv_source_layer(*model, il);
        const uint32_t rope_source = kv_source != TRIATTENTION_LAYER_NONE ? kv_source : il;

        layer.kv_source_layer = kv_source;
        layer.head_dim        = hparams.n_embd_head_k(rope_source);
        layer.rope_dim        = hparams.n_rot(il) > 0 ? hparams.n_rot(il) : hparams.n_rot(rope_source);
        if (layer.rope_dim == 0) {
            layer.rope_dim = layer.head_dim;
        }
        if ((layer.rope_dim % 2) != 0 || layer.head_dim < layer.rope_dim) {
            triattention_model_params_clear(out);
            return false;
        }
        layer.rope_offset    = layer.head_dim - layer.rope_dim;
        layer.num_attn_heads = hparams.n_head(il);
        layer.num_kv_heads   = kv_source != TRIATTENTION_LAYER_NONE ? hparams.n_head_kv(kv_source) : hparams.n_head_kv(il);
        layer.num_kv_groups  = layer.num_kv_heads > 0 ? layer.num_attn_heads / layer.num_kv_heads : 0;
        layer.rope_style     = rope_style;
        layer.freq_count     = layer.rope_dim / 2;
        layer.n_ctx_orig     = hparams.n_ctx_orig_yarn;

        layer.rope_theta       = (double) model->get_rope_freq_base(*cparams, rope_source);
        layer.rope_freq_scale  = model->get_rope_freq_scale(*cparams, rope_source);
        layer.rope_ext_factor  = cparams->yarn_ext_factor;
        layer.rope_attn_factor = cparams->yarn_attn_factor;
        layer.rope_beta_fast   = cparams->yarn_beta_fast;
        layer.rope_beta_slow   = cparams->yarn_beta_slow;

        if (layer.freq_count == 0) {
            continue;
        }

        std::vector<float> freq_factors_host;
        const float * freq_factors = nullptr;
        uint32_t freq_factor_count = 0;
        if (model->layers[rope_source].rope_freqs) {
            ggml_tensor * rope_freqs = model->layers[rope_source].rope_freqs;
            freq_factor_count = (uint32_t) rope_freqs->ne[0];
            const bool is_host = rope_freqs->buffer == nullptr || ggml_backend_buffer_is_host(rope_freqs->buffer);
            if (is_host) {
                freq_factors = (const float *) rope_freqs->data;
            } else {
                freq_factors_host.resize(freq_factor_count);
                ggml_backend_tensor_get(rope_freqs, freq_factors_host.data(), 0, sizeof(float) * freq_factor_count);
                freq_factors = freq_factors_host.data();
            }
        }

        triattention_rope_params rope_params = {
            /*.n_dims            =*/ layer.rope_dim,
            /*.n_ctx_orig        =*/ layer.n_ctx_orig,
            /*.freq_base         =*/ (float) layer.rope_theta,
            /*.freq_scale        =*/ layer.rope_freq_scale,
            /*.ext_factor        =*/ layer.rope_ext_factor,
            /*.attn_factor       =*/ layer.rope_attn_factor,
            /*.beta_fast         =*/ layer.rope_beta_fast,
            /*.beta_slow         =*/ layer.rope_beta_slow,
            /*.freq_factors      =*/ freq_factors,
            /*.freq_factor_count =*/ freq_factor_count,
        };

        layer.omega = new float[layer.freq_count];
        layer.freq_scale_sq = new float[layer.freq_count];
        if (!triattention_build_rope_arrays(layer.omega, layer.freq_scale_sq, layer.freq_count, &rope_params)) {
            triattention_model_params_clear(out);
            return false;
        }
    }

    out->head_dim         = 0;
    out->rope_dim         = 0;
    out->num_attn_heads   = 0;
    out->num_kv_heads     = 0;
    out->rope_style       = rope_style;
    out->n_ctx_orig       = hparams.n_ctx_orig_yarn;
    out->rope_theta       = out->num_layers > 0 ? out->layers[0].rope_theta : 10000.0;
    out->rope_freq_scale  = out->num_layers > 0 ? out->layers[0].rope_freq_scale : 1.0f;
    out->rope_ext_factor  = cparams->yarn_ext_factor;
    out->rope_attn_factor = cparams->yarn_attn_factor;
    out->rope_beta_fast   = cparams->yarn_beta_fast;
    out->rope_beta_slow   = cparams->yarn_beta_slow;
    out->max_head_dim     = 0;
    out->max_rope_dim     = 0;
    out->max_freq_count   = 0;
    out->heterogeneous_layout = false;

    if (out->num_layers > 0) {
        const triattention_layer_params & base = out->layers[0];
        for (uint32_t il = 0; il < out->num_layers; ++il) {
            const triattention_layer_params & layer = out->layers[il];
            out->head_dim       = std::max(out->head_dim, layer.head_dim);
            out->rope_dim       = std::max(out->rope_dim, layer.rope_dim);
            out->num_attn_heads = std::max(out->num_attn_heads, layer.num_attn_heads);
            out->num_kv_heads   = std::max(out->num_kv_heads, layer.num_kv_heads);
            out->max_head_dim   = std::max(out->max_head_dim, layer.head_dim);
            out->max_rope_dim   = std::max(out->max_rope_dim, layer.rope_dim);
            out->max_freq_count = std::max(out->max_freq_count, layer.freq_count);
            if (il > 0 && !triattention_layer_params_equal(base, layer)) {
                out->heterogeneous_layout = true;
            }
        }
    }

    return true;
}

void triattention_calibration_free(triattention_calibration * cal) {
    if (!cal) {
        return;
    }

    for (uint32_t i = 0; i < cal->n_sampled; ++i) {
        triattention_free_head_stats(&cal->head_stats[i]);
    }

    delete[] cal->omega;
    delete[] cal->freq_scale_sq;
    delete[] cal->sampled_layer;
    delete[] cal->sampled_head;
    delete[] cal->head_stats;
    triattention_clear_layer_params_array(cal->layers, cal->num_layers);
    delete cal;
}

static bool triattention_read_legacy_header(
    triattention_binary_reader & reader,
    triattention_calibration * cal) {
    bool ok = true;
    ok = ok && reader.read(&cal->head_dim,       sizeof(uint32_t));
    ok = ok && reader.read(&cal->num_layers,     sizeof(uint32_t));
    ok = ok && reader.read(&cal->num_attn_heads, sizeof(uint32_t));
    ok = ok && reader.read(&cal->num_kv_heads,   sizeof(uint32_t));
    ok = ok && reader.read(&cal->rope_theta,     sizeof(double));
    ok = ok && reader.read(&cal->rope_style,     sizeof(uint32_t));
    ok = ok && reader.read(&cal->n_sampled,      sizeof(uint32_t));
    ok = ok && reader.read(&cal->freq_count,     sizeof(uint32_t));
    return ok;
}

static bool triattention_read_model_name(
    triattention_binary_reader & reader,
    triattention_calibration * cal) {
    uint32_t name_len = 0;
    if (!reader.read(&name_len, sizeof(name_len)) || name_len == 0 || name_len > sizeof(cal->model_name)) {
        fprintf(stderr, "[TriAttention] ERROR: invalid model name length %u in %s\n", name_len, reader.source);
        return false;
    }

    if (!reader.read(cal->model_name, name_len)) {
        fprintf(stderr, "[TriAttention] ERROR: truncated model name in %s\n", reader.source);
        return false;
    }

    cal->model_name[sizeof(cal->model_name) - 1] = '\0';
    return true;
}

static bool triattention_read_v3_layers(
    triattention_binary_reader & reader,
    triattention_calibration * cal) {
    uint32_t layer_count = 0;
    if (!reader.read(&layer_count, sizeof(layer_count)) || layer_count != cal->num_layers) {
        fprintf(stderr, "[TriAttention] ERROR: invalid v3 layer count in %s\n", reader.source);
        return false;
    }

    cal->layers = new triattention_layer_params[cal->num_layers];
    memset(cal->layers, 0, sizeof(triattention_layer_params) * cal->num_layers);

    for (uint32_t il = 0; il < cal->num_layers; ++il) {
        triattention_layer_params & layer = cal->layers[il];
        uint32_t has_explicit_rope = 0;

        bool ok = true;
        ok = ok && reader.read(&layer.head_dim,         sizeof(uint32_t));
        ok = ok && reader.read(&layer.rope_dim,         sizeof(uint32_t));
        ok = ok && reader.read(&layer.rope_offset,      sizeof(uint32_t));
        ok = ok && reader.read(&layer.num_attn_heads,   sizeof(uint32_t));
        ok = ok && reader.read(&layer.num_kv_heads,     sizeof(uint32_t));
        ok = ok && reader.read(&layer.kv_source_layer,  sizeof(uint32_t));
        ok = ok && reader.read(&layer.rope_style,       sizeof(uint32_t));
        ok = ok && reader.read(&layer.n_ctx_orig,       sizeof(uint32_t));
        ok = ok && reader.read(&layer.rope_theta,       sizeof(double));
        ok = ok && reader.read(&layer.rope_freq_scale,  sizeof(float));
        ok = ok && reader.read(&layer.rope_ext_factor,  sizeof(float));
        ok = ok && reader.read(&layer.rope_attn_factor, sizeof(float));
        ok = ok && reader.read(&layer.rope_beta_fast,   sizeof(float));
        ok = ok && reader.read(&layer.rope_beta_slow,   sizeof(float));
        ok = ok && reader.read(&layer.freq_count,       sizeof(uint32_t));
        ok = ok && reader.read(&has_explicit_rope,      sizeof(uint32_t));
        if (!ok) {
            fprintf(stderr, "[TriAttention] ERROR: truncated v3 layer %u in %s\n", il, reader.source);
            return false;
        }

        layer.num_kv_groups = layer.num_kv_heads > 0 ? layer.num_attn_heads / layer.num_kv_heads : 0;

        if (layer.rope_dim != layer.freq_count * 2 || layer.head_dim < layer.rope_dim ||
            layer.rope_offset > layer.head_dim - layer.rope_dim) {
            fprintf(stderr, "[TriAttention] ERROR: invalid v3 layer geometry in %s\n", reader.source);
            return false;
        }

        if (has_explicit_rope) {
            layer.omega = new float[layer.freq_count];
            layer.freq_scale_sq = new float[layer.freq_count];
            ok = true;
            ok = ok && reader.read(layer.omega,         sizeof(float) * layer.freq_count);
            ok = ok && reader.read(layer.freq_scale_sq, sizeof(float) * layer.freq_count);
            if (!ok) {
                fprintf(stderr, "[TriAttention] ERROR: truncated v3 rope arrays in %s\n", reader.source);
                return false;
            }
        }
    }

    triattention_update_summary_from_layers(cal, cal->layers, cal->num_layers);
    if (!cal->heterogeneous_layout && cal->num_layers > 0 && cal->layers[0].omega && cal->layers[0].freq_scale_sq) {
        cal->omega = new float[cal->layers[0].freq_count];
        cal->freq_scale_sq = new float[cal->layers[0].freq_count];
        memcpy(cal->omega, cal->layers[0].omega, sizeof(float) * cal->layers[0].freq_count);
        memcpy(cal->freq_scale_sq, cal->layers[0].freq_scale_sq, sizeof(float) * cal->layers[0].freq_count);
    }

    return true;
}

static bool triattention_write_legacy_header(
    triattention_binary_writer & writer,
    const triattention_calibration * cal,
    uint32_t version,
    uint32_t name_len) {
    const uint32_t magic = TRIATTENTION_MAGIC;

    bool ok = true;
    ok = ok && writer.write(&magic,               sizeof(magic));
    ok = ok && writer.write(&version,             sizeof(version));
    ok = ok && writer.write(&cal->head_dim,       sizeof(uint32_t));
    ok = ok && writer.write(&cal->num_layers,     sizeof(uint32_t));
    ok = ok && writer.write(&cal->num_attn_heads, sizeof(uint32_t));
    ok = ok && writer.write(&cal->num_kv_heads,   sizeof(uint32_t));
    ok = ok && writer.write(&cal->rope_theta,     sizeof(double));
    ok = ok && writer.write(&cal->rope_style,     sizeof(uint32_t));
    ok = ok && writer.write(&cal->n_sampled,      sizeof(uint32_t));
    ok = ok && writer.write(&cal->freq_count,     sizeof(uint32_t));
    ok = ok && writer.write(&name_len,            sizeof(name_len));
    ok = ok && writer.write(cal->model_name,      name_len);
    return ok;
}

static bool triattention_read_legacy_header(FILE * f, triattention_calibration * cal) {
    bool ok = true;
    ok = ok && fread(&cal->head_dim,       sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_layers,     sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_attn_heads, sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_kv_heads,   sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->rope_theta,     sizeof(double),   1, f) == 1;
    ok = ok && fread(&cal->rope_style,     sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->n_sampled,      sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->freq_count,     sizeof(uint32_t), 1, f) == 1;
    return ok;
}

static bool triattention_read_model_name(FILE * f, triattention_calibration * cal, const char * path) {
    uint32_t name_len = 0;
    if (fread(&name_len, sizeof(name_len), 1, f) != 1 || name_len == 0 || name_len > sizeof(cal->model_name)) {
        fprintf(stderr, "[TriAttention] ERROR: invalid model name length %u in %s\n", name_len, path);
        return false;
    }

    if (fread(cal->model_name, 1, name_len, f) != name_len) {
        fprintf(stderr, "[TriAttention] ERROR: truncated model name in %s\n", path);
        return false;
    }

    cal->model_name[sizeof(cal->model_name) - 1] = '\0';
    return true;
}

static bool triattention_build_uniform_layers(triattention_calibration * cal) {
    if (cal->num_layers == 0 || cal->freq_count != cal->head_dim / 2) {
        return false;
    }

    cal->layers = new triattention_layer_params[cal->num_layers];
    memset(cal->layers, 0, sizeof(triattention_layer_params) * cal->num_layers);
    for (uint32_t il = 0; il < cal->num_layers; ++il) {
        triattention_layer_params & layer = cal->layers[il];
        layer.head_dim        = cal->head_dim;
        layer.rope_dim        = cal->head_dim;
        layer.rope_offset     = 0;
        layer.num_attn_heads  = cal->num_attn_heads;
        layer.num_kv_heads    = cal->num_kv_heads;
        layer.num_kv_groups   = cal->num_kv_groups;
        layer.kv_source_layer = il;
        layer.rope_style      = cal->rope_style;
        layer.freq_count      = cal->freq_count;
        layer.n_ctx_orig      = 0;
        layer.rope_theta      = cal->rope_theta;
        layer.rope_freq_scale = 1.0f;
        layer.rope_ext_factor = 0.0f;
        layer.rope_attn_factor = 1.0f;
        layer.rope_beta_fast   = 32.0f;
        layer.rope_beta_slow   = 1.0f;

        if (cal->omega && cal->freq_scale_sq) {
            layer.omega = new float[layer.freq_count];
            layer.freq_scale_sq = new float[layer.freq_count];
            memcpy(layer.omega, cal->omega, sizeof(float) * layer.freq_count);
            memcpy(layer.freq_scale_sq, cal->freq_scale_sq, sizeof(float) * layer.freq_count);
        }
    }

    triattention_update_summary_from_layers(cal, cal->layers, cal->num_layers);
    return true;
}

static bool triattention_read_v3_layers(FILE * f, triattention_calibration * cal, const char * path) {
    uint32_t layer_count = 0;
    if (fread(&layer_count, sizeof(layer_count), 1, f) != 1 || layer_count != cal->num_layers) {
        fprintf(stderr, "[TriAttention] ERROR: invalid v3 layer count in %s\n", path);
        return false;
    }

    cal->layers = new triattention_layer_params[cal->num_layers];
    memset(cal->layers, 0, sizeof(triattention_layer_params) * cal->num_layers);

    for (uint32_t il = 0; il < cal->num_layers; ++il) {
        triattention_layer_params & layer = cal->layers[il];
        uint32_t has_explicit_rope = 0;

        bool ok = true;
        ok = ok && fread(&layer.head_dim,         sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&layer.rope_dim,         sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&layer.rope_offset,      sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&layer.num_attn_heads,   sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&layer.num_kv_heads,     sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&layer.kv_source_layer,  sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&layer.rope_style,       sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&layer.n_ctx_orig,       sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&layer.rope_theta,       sizeof(double),   1, f) == 1;
        ok = ok && fread(&layer.rope_freq_scale,  sizeof(float),    1, f) == 1;
        ok = ok && fread(&layer.rope_ext_factor,  sizeof(float),    1, f) == 1;
        ok = ok && fread(&layer.rope_attn_factor, sizeof(float),    1, f) == 1;
        ok = ok && fread(&layer.rope_beta_fast,   sizeof(float),    1, f) == 1;
        ok = ok && fread(&layer.rope_beta_slow,   sizeof(float),    1, f) == 1;
        ok = ok && fread(&layer.freq_count,       sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&has_explicit_rope,      sizeof(uint32_t), 1, f) == 1;
        if (!ok) {
            fprintf(stderr, "[TriAttention] ERROR: truncated v3 layer %u in %s\n", il, path);
            return false;
        }

        layer.num_kv_groups = layer.num_kv_heads > 0 ? layer.num_attn_heads / layer.num_kv_heads : 0;

        if (layer.rope_dim != layer.freq_count * 2 || layer.head_dim < layer.rope_dim ||
            layer.rope_offset > layer.head_dim - layer.rope_dim) {
            fprintf(stderr, "[TriAttention] ERROR: invalid v3 layer geometry in %s\n", path);
            return false;
        }

        if (has_explicit_rope) {
            layer.omega = new float[layer.freq_count];
            layer.freq_scale_sq = new float[layer.freq_count];
            ok = true;
            ok = ok && fread(layer.omega,         sizeof(float), layer.freq_count, f) == layer.freq_count;
            ok = ok && fread(layer.freq_scale_sq, sizeof(float), layer.freq_count, f) == layer.freq_count;
            if (!ok) {
                fprintf(stderr, "[TriAttention] ERROR: truncated v3 rope arrays in %s\n", path);
                return false;
            }
        }
    }

    triattention_update_summary_from_layers(cal, cal->layers, cal->num_layers);
    if (!cal->heterogeneous_layout && cal->num_layers > 0 && cal->layers[0].omega && cal->layers[0].freq_scale_sq) {
        cal->omega = new float[cal->layers[0].freq_count];
        cal->freq_scale_sq = new float[cal->layers[0].freq_count];
        memcpy(cal->omega, cal->layers[0].omega, sizeof(float) * cal->layers[0].freq_count);
        memcpy(cal->freq_scale_sq, cal->layers[0].freq_scale_sq, sizeof(float) * cal->layers[0].freq_count);
    }

    return true;
}

triattention_calibration * triattention_calibration_load_from_buffer(
    const void * data,
    size_t size,
    bool verbose,
    const char * source_name) {
    if (!data || size < sizeof(uint32_t) * 2) {
        fprintf(stderr, "[TriAttention] ERROR: calibration buffer is empty or truncated (%s)\n",
                source_name ? source_name : "<memory>");
        return nullptr;
    }

    triattention_binary_reader reader = {};
    reader.data = static_cast<const uint8_t *>(data);
    reader.size = size;
    reader.source = source_name ? source_name : "<memory>";

    uint32_t magic = 0;
    if (!reader.read(&magic, sizeof(magic)) || magic != TRIATTENTION_MAGIC) {
        fprintf(stderr, "[TriAttention] ERROR: invalid magic in %s (got 0x%08x, expected 0x%08x)\n",
                reader.source, magic, TRIATTENTION_MAGIC);
        return nullptr;
    }

    uint32_t version = 0;
    if (!reader.read(&version, sizeof(version)) ||
        (version != TRIATTENTION_VERSION_V1 && version != TRIATTENTION_VERSION_V2 && version != TRIATTENTION_VERSION)) {
        fprintf(stderr, "[TriAttention] ERROR: unsupported version %u in %s\n", version, reader.source);
        return nullptr;
    }

    auto * cal = new triattention_calibration();
    memset(cal, 0, sizeof(*cal));
    cal->version = version;

    if (!triattention_read_legacy_header(reader, cal)) {
        fprintf(stderr, "[TriAttention] ERROR: truncated header in %s\n", reader.source);
        triattention_calibration_free(cal);
        return nullptr;
    }

    if (!triattention_read_model_name(reader, cal)) {
        triattention_calibration_free(cal);
        return nullptr;
    }

    if (version == TRIATTENTION_VERSION_V2) {
        cal->omega = new float[cal->freq_count];
        cal->freq_scale_sq = new float[cal->freq_count];

        bool ok = true;
        ok = ok && reader.read(cal->omega,         sizeof(float) * cal->freq_count);
        ok = ok && reader.read(cal->freq_scale_sq, sizeof(float) * cal->freq_count);
        if (!ok) {
            fprintf(stderr, "[TriAttention] ERROR: truncated v2 rope arrays in %s\n", reader.source);
            triattention_calibration_free(cal);
            return nullptr;
        }
    }

    if (version == TRIATTENTION_VERSION) {
        if (!triattention_read_v3_layers(reader, cal)) {
            triattention_calibration_free(cal);
            return nullptr;
        }
    } else if (!triattention_build_uniform_layers(cal)) {
        fprintf(stderr, "[TriAttention] ERROR: invalid legacy calibration geometry in %s\n", reader.source);
        triattention_calibration_free(cal);
        return nullptr;
    }

    if (cal->num_attn_heads == 0 || cal->num_layers == 0) {
        fprintf(stderr, "[TriAttention] ERROR: invalid model dimensions in %s\n", reader.source);
        triattention_calibration_free(cal);
        return nullptr;
    }

    cal->sampled_layer = new uint32_t[cal->n_sampled];
    cal->sampled_head  = new uint32_t[cal->n_sampled];
    cal->head_stats    = new triattention_head_stats[cal->n_sampled];
    memset(cal->head_stats, 0, sizeof(triattention_head_stats) * cal->n_sampled);

    for (uint32_t h = 0; h < cal->n_sampled; ++h) {
        auto & hs = cal->head_stats[h];
        bool ok = true;
        ok = ok && reader.read(&cal->sampled_layer[h], sizeof(uint32_t));
        ok = ok && reader.read(&cal->sampled_head[h],  sizeof(uint32_t));
        if (!ok || cal->sampled_layer[h] >= cal->num_layers) {
            fprintf(stderr, "[TriAttention] ERROR: truncated or invalid head entry %u in %s\n", h, reader.source);
            triattention_calibration_free(cal);
            return nullptr;
        }

        const uint32_t freq_count = cal->layers[cal->sampled_layer[h]].freq_count;
        hs.q_mean_real = new float[freq_count];
        hs.q_mean_imag = new float[freq_count];
        hs.q_abs_mean  = new float[freq_count];
        hs.r_f         = new float[freq_count];

        ok = true;
        ok = ok && reader.read(hs.q_mean_real, sizeof(float) * freq_count);
        ok = ok && reader.read(hs.q_mean_imag, sizeof(float) * freq_count);
        ok = ok && reader.read(hs.q_abs_mean,  sizeof(float) * freq_count);
        ok = ok && reader.read(hs.r_f,         sizeof(float) * freq_count);
        if (!ok) {
            fprintf(stderr, "[TriAttention] ERROR: truncated stats for head %u in %s\n", h, reader.source);
            triattention_calibration_free(cal);
            return nullptr;
        }
    }

    if (verbose) {
        fprintf(stderr, "[TriAttention] Loaded calibration: model=%s, version=%u, layers=%u, sampled=%u%s\n",
                cal->model_name, cal->version, cal->num_layers, cal->n_sampled,
                cal->heterogeneous_layout ? ", heterogeneous=yes" : "");
    }

    return cal;
}

triattention_calibration * triattention_calibration_load(const char * path, bool verbose) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[TriAttention] ERROR: cannot open calibration file: %s\n", path);
        return nullptr;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "[TriAttention] ERROR: failed to seek calibration file: %s\n", path);
        fclose(f);
        return nullptr;
    }

    const long size_long = ftell(f);
    if (size_long < 0) {
        fprintf(stderr, "[TriAttention] ERROR: failed to determine calibration size: %s\n", path);
        fclose(f);
        return nullptr;
    }
    rewind(f);

    std::vector<uint8_t> buffer((size_t) size_long);
    if (!buffer.empty() && fread(buffer.data(), 1, buffer.size(), f) != buffer.size()) {
        fprintf(stderr, "[TriAttention] ERROR: failed to read calibration file: %s\n", path);
        fclose(f);
        return nullptr;
    }

    fclose(f);
    return triattention_calibration_load_from_buffer(buffer.data(), buffer.size(), verbose, path);
}

static bool triattention_write_legacy_header(FILE * f, const triattention_calibration * cal, uint32_t version, uint32_t name_len) {
    const uint32_t magic = TRIATTENTION_MAGIC;

    bool ok = true;
    ok = ok && fwrite(&magic,               sizeof(magic),               1, f) == 1;
    ok = ok && fwrite(&version,             sizeof(version),             1, f) == 1;
    ok = ok && fwrite(&cal->head_dim,       sizeof(uint32_t),            1, f) == 1;
    ok = ok && fwrite(&cal->num_layers,     sizeof(uint32_t),            1, f) == 1;
    ok = ok && fwrite(&cal->num_attn_heads, sizeof(uint32_t),            1, f) == 1;
    ok = ok && fwrite(&cal->num_kv_heads,   sizeof(uint32_t),            1, f) == 1;
    ok = ok && fwrite(&cal->rope_theta,     sizeof(double),              1, f) == 1;
    ok = ok && fwrite(&cal->rope_style,     sizeof(uint32_t),            1, f) == 1;
    ok = ok && fwrite(&cal->n_sampled,      sizeof(uint32_t),            1, f) == 1;
    ok = ok && fwrite(&cal->freq_count,     sizeof(uint32_t),            1, f) == 1;
    ok = ok && fwrite(&name_len,            sizeof(name_len),            1, f) == 1;
    ok = ok && fwrite(cal->model_name,      1,                           name_len, f) == name_len;
    return ok;
}

bool triattention_calibration_save_to_buffer(
    const triattention_calibration * cal,
    std::vector<uint8_t> & out) {
    if (!cal) {
        return false;
    }

    out.clear();
    triattention_binary_writer writer = { &out };

    const uint32_t version = cal->version == 0 ? TRIATTENTION_VERSION : cal->version;
    const uint32_t name_len = (uint32_t) strnlen(cal->model_name, sizeof(cal->model_name) - 1) + 1;

    bool ok = triattention_write_legacy_header(writer, cal, version, name_len);

    if (version == TRIATTENTION_VERSION_V2) {
        ok = ok && cal->omega != nullptr && cal->freq_scale_sq != nullptr;
        ok = ok && writer.write(cal->omega,         sizeof(float) * cal->freq_count);
        ok = ok && writer.write(cal->freq_scale_sq, sizeof(float) * cal->freq_count);
    } else if (version >= TRIATTENTION_VERSION) {
        const uint32_t layer_count = cal->num_layers;
        ok = ok && writer.write(&layer_count, sizeof(layer_count));
        for (uint32_t il = 0; ok && il < cal->num_layers; ++il) {
            const triattention_layer_params & layer = cal->layers[il];
            const uint32_t has_explicit_rope = layer.omega && layer.freq_scale_sq ? 1u : 0u;
            ok = ok && writer.write(&layer.head_dim,         sizeof(uint32_t));
            ok = ok && writer.write(&layer.rope_dim,         sizeof(uint32_t));
            ok = ok && writer.write(&layer.rope_offset,      sizeof(uint32_t));
            ok = ok && writer.write(&layer.num_attn_heads,   sizeof(uint32_t));
            ok = ok && writer.write(&layer.num_kv_heads,     sizeof(uint32_t));
            ok = ok && writer.write(&layer.kv_source_layer,  sizeof(uint32_t));
            ok = ok && writer.write(&layer.rope_style,       sizeof(uint32_t));
            ok = ok && writer.write(&layer.n_ctx_orig,       sizeof(uint32_t));
            ok = ok && writer.write(&layer.rope_theta,       sizeof(double));
            ok = ok && writer.write(&layer.rope_freq_scale,  sizeof(float));
            ok = ok && writer.write(&layer.rope_ext_factor,  sizeof(float));
            ok = ok && writer.write(&layer.rope_attn_factor, sizeof(float));
            ok = ok && writer.write(&layer.rope_beta_fast,   sizeof(float));
            ok = ok && writer.write(&layer.rope_beta_slow,   sizeof(float));
            ok = ok && writer.write(&layer.freq_count,       sizeof(uint32_t));
            ok = ok && writer.write(&has_explicit_rope,      sizeof(uint32_t));
            if (has_explicit_rope) {
                ok = ok && writer.write(layer.omega,         sizeof(float) * layer.freq_count);
                ok = ok && writer.write(layer.freq_scale_sq, sizeof(float) * layer.freq_count);
            }
        }
    }

    for (uint32_t h = 0; ok && h < cal->n_sampled; ++h) {
        const uint32_t layer_idx = cal->sampled_layer[h];
        const uint32_t freq_count = cal->layers[layer_idx].freq_count;
        const triattention_head_stats & hs = cal->head_stats[h];
        ok = ok && hs.q_mean_real != nullptr && hs.q_mean_imag != nullptr && hs.q_abs_mean != nullptr && hs.r_f != nullptr;
        ok = ok && writer.write(&cal->sampled_layer[h], sizeof(uint32_t));
        ok = ok && writer.write(&cal->sampled_head[h],  sizeof(uint32_t));
        ok = ok && writer.write(hs.q_mean_real, sizeof(float) * freq_count);
        ok = ok && writer.write(hs.q_mean_imag, sizeof(float) * freq_count);
        ok = ok && writer.write(hs.q_abs_mean,  sizeof(float) * freq_count);
        ok = ok && writer.write(hs.r_f,         sizeof(float) * freq_count);
    }

    return ok;
}

bool triattention_calibration_save(const char * path, const triattention_calibration * cal) {
    if (!path || !cal) {
        return false;
    }

    std::vector<uint8_t> buffer;
    if (!triattention_calibration_save_to_buffer(cal, buffer)) {
        fprintf(stderr, "[TriAttention] ERROR: failed while serializing calibration file: %s\n", path);
        return false;
    }

    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[TriAttention] ERROR: cannot write calibration file: %s\n", path);
        return false;
    }

    const bool ok = buffer.empty() || fwrite(buffer.data(), 1, buffer.size(), f) == buffer.size();
    fclose(f);

    if (!ok) {
        fprintf(stderr, "[TriAttention] ERROR: failed while writing calibration file: %s\n", path);
        return false;
    }

    return true;
}

triattention_calibration * triattention_calibration_subset(
    const triattention_calibration * cal,
    const uint32_t * sampled_layers,
    uint32_t n_sampled_layers) {
    if (!cal) {
        return nullptr;
    }

    if (!sampled_layers || n_sampled_layers == 0) {
        triattention_calibration * copy = new triattention_calibration();
        memset(copy, 0, sizeof(*copy));
        *copy = *cal;
        copy->omega = nullptr;
        copy->freq_scale_sq = nullptr;
        copy->layers = nullptr;
        copy->sampled_layer = nullptr;
        copy->sampled_head = nullptr;
        copy->head_stats = nullptr;

        if (cal->omega && cal->freq_count > 0) {
            copy->omega = new float[cal->freq_count];
            memcpy(copy->omega, cal->omega, sizeof(float) * cal->freq_count);
        }
        if (cal->freq_scale_sq && cal->freq_count > 0) {
            copy->freq_scale_sq = new float[cal->freq_count];
            memcpy(copy->freq_scale_sq, cal->freq_scale_sq, sizeof(float) * cal->freq_count);
        }

        copy->layers = new triattention_layer_params[cal->num_layers];
        memset(copy->layers, 0, sizeof(triattention_layer_params) * cal->num_layers);
        for (uint32_t il = 0; il < cal->num_layers; ++il) {
            triattention_copy_layer_params(&copy->layers[il], &cal->layers[il]);
        }

        copy->sampled_layer = new uint32_t[cal->n_sampled];
        copy->sampled_head  = new uint32_t[cal->n_sampled];
        copy->head_stats    = new triattention_head_stats[cal->n_sampled];
        memset(copy->head_stats, 0, sizeof(triattention_head_stats) * cal->n_sampled);
        for (uint32_t h = 0; h < cal->n_sampled; ++h) {
            const uint32_t freq_count = cal->layers[cal->sampled_layer[h]].freq_count;
            copy->sampled_layer[h] = cal->sampled_layer[h];
            copy->sampled_head[h]  = cal->sampled_head[h];
            copy->head_stats[h].q_mean_real = new float[freq_count];
            copy->head_stats[h].q_mean_imag = new float[freq_count];
            copy->head_stats[h].q_abs_mean  = new float[freq_count];
            copy->head_stats[h].r_f         = new float[freq_count];
            memcpy(copy->head_stats[h].q_mean_real, cal->head_stats[h].q_mean_real, sizeof(float) * freq_count);
            memcpy(copy->head_stats[h].q_mean_imag, cal->head_stats[h].q_mean_imag, sizeof(float) * freq_count);
            memcpy(copy->head_stats[h].q_abs_mean,  cal->head_stats[h].q_abs_mean,  sizeof(float) * freq_count);
            memcpy(copy->head_stats[h].r_f,         cal->head_stats[h].r_f,         sizeof(float) * freq_count);
        }
        return copy;
    }

    std::vector<bool> keep_layer(cal->num_layers, false);
    for (uint32_t i = 0; i < n_sampled_layers; ++i) {
        if (sampled_layers[i] < cal->num_layers) {
            keep_layer[sampled_layers[i]] = true;
        }
    }

    uint32_t kept = 0;
    for (uint32_t h = 0; h < cal->n_sampled; ++h) {
        kept += keep_layer[cal->sampled_layer[h]] ? 1u : 0u;
    }

    auto * out = new triattention_calibration();
    memset(out, 0, sizeof(*out));
    *out = *cal;
    out->omega = nullptr;
    out->freq_scale_sq = nullptr;
    out->layers = new triattention_layer_params[cal->num_layers];
    memset(out->layers, 0, sizeof(triattention_layer_params) * cal->num_layers);
    for (uint32_t il = 0; il < cal->num_layers; ++il) {
        triattention_copy_layer_params(&out->layers[il], &cal->layers[il]);
    }

    out->sampled_layer = new uint32_t[kept];
    out->sampled_head  = new uint32_t[kept];
    out->head_stats    = new triattention_head_stats[kept];
    memset(out->head_stats, 0, sizeof(triattention_head_stats) * kept);
    out->n_sampled = kept;

    uint32_t out_idx = 0;
    for (uint32_t h = 0; h < cal->n_sampled; ++h) {
        if (!keep_layer[cal->sampled_layer[h]]) {
            continue;
        }

        const uint32_t freq_count = cal->layers[cal->sampled_layer[h]].freq_count;
        out->sampled_layer[out_idx] = cal->sampled_layer[h];
        out->sampled_head[out_idx]  = cal->sampled_head[h];
        out->head_stats[out_idx].q_mean_real = new float[freq_count];
        out->head_stats[out_idx].q_mean_imag = new float[freq_count];
        out->head_stats[out_idx].q_abs_mean  = new float[freq_count];
        out->head_stats[out_idx].r_f         = new float[freq_count];
        memcpy(out->head_stats[out_idx].q_mean_real, cal->head_stats[h].q_mean_real, sizeof(float) * freq_count);
        memcpy(out->head_stats[out_idx].q_mean_imag, cal->head_stats[h].q_mean_imag, sizeof(float) * freq_count);
        memcpy(out->head_stats[out_idx].q_abs_mean,  cal->head_stats[h].q_abs_mean,  sizeof(float) * freq_count);
        memcpy(out->head_stats[out_idx].r_f,         cal->head_stats[h].r_f,         sizeof(float) * freq_count);
        ++out_idx;
    }

    if (!out->heterogeneous_layout && out->num_layers > 0 && out->layers[0].omega && out->layers[0].freq_scale_sq) {
        out->omega = new float[out->layers[0].freq_count];
        out->freq_scale_sq = new float[out->layers[0].freq_count];
        memcpy(out->omega, out->layers[0].omega, sizeof(float) * out->layers[0].freq_count);
        memcpy(out->freq_scale_sq, out->layers[0].freq_scale_sq, sizeof(float) * out->layers[0].freq_count);
    }

    return out;
}

bool triattention_calibration_validate(
    const triattention_calibration * cal,
    const triattention_model_params * model,
    bool warn_rope_theta) {
    if (!cal || !model || cal->num_layers != model->num_layers || !cal->layers || !model->layers) {
        return false;
    }

    for (uint32_t il = 0; il < model->num_layers; ++il) {
        const triattention_layer_params & lhs = cal->layers[il];
        const triattention_layer_params & rhs = model->layers[il];

        if (lhs.head_dim != rhs.head_dim) {
            fprintf(stderr, "[TriAttention] ERROR: layer %u head_dim mismatch (calibration=%u, model=%u)\n",
                    il, lhs.head_dim, rhs.head_dim);
            return false;
        }
        if (lhs.rope_dim != rhs.rope_dim) {
            fprintf(stderr, "[TriAttention] ERROR: layer %u rope_dim mismatch (calibration=%u, model=%u)\n",
                    il, lhs.rope_dim, rhs.rope_dim);
            return false;
        }
        if (lhs.rope_offset != rhs.rope_offset) {
            fprintf(stderr, "[TriAttention] ERROR: layer %u rope_offset mismatch (calibration=%u, model=%u)\n",
                    il, lhs.rope_offset, rhs.rope_offset);
            return false;
        }
        if (lhs.num_attn_heads != rhs.num_attn_heads) {
            fprintf(stderr, "[TriAttention] ERROR: layer %u attention head mismatch (calibration=%u, model=%u)\n",
                    il, lhs.num_attn_heads, rhs.num_attn_heads);
            return false;
        }
        if (lhs.num_kv_heads != rhs.num_kv_heads) {
            fprintf(stderr, "[TriAttention] ERROR: layer %u KV head mismatch (calibration=%u, model=%u)\n",
                    il, lhs.num_kv_heads, rhs.num_kv_heads);
            return false;
        }
        if (lhs.rope_style != rhs.rope_style) {
            fprintf(stderr, "[TriAttention] ERROR: layer %u rope style mismatch (calibration=%u, model=%u)\n",
                    il, lhs.rope_style, rhs.rope_style);
            return false;
        }
        if (warn_rope_theta && fabs(lhs.rope_theta - rhs.rope_theta) / fmax(lhs.rope_theta, 1.0) > 0.01) {
            fprintf(stderr, "[TriAttention] WARNING: layer %u rope_theta mismatch (calibration=%.1f, model=%.1f)\n",
                    il, lhs.rope_theta, rhs.rope_theta);
        }
    }

    return true;
}

triattention_calibration * triattention_calibration_create_fallback(
    const triattention_model_params * model,
    const uint32_t * sampled_layers,
    uint32_t n_sampled_layers) {
    if (!model || model->num_layers == 0 || !model->layers) {
        return nullptr;
    }

    std::vector<bool> keep_layer(model->num_layers, true);
    if (sampled_layers && n_sampled_layers > 0) {
        std::fill(keep_layer.begin(), keep_layer.end(), false);
        for (uint32_t i = 0; i < n_sampled_layers; ++i) {
            if (sampled_layers[i] < model->num_layers) {
                keep_layer[sampled_layers[i]] = true;
            }
        }
    }

    auto * cal = new triattention_calibration();
    memset(cal, 0, sizeof(*cal));

    cal->version    = TRIATTENTION_VERSION;
    cal->num_layers = model->num_layers;
    cal->layers     = new triattention_layer_params[cal->num_layers];
    memset(cal->layers, 0, sizeof(triattention_layer_params) * cal->num_layers);
    for (uint32_t il = 0; il < cal->num_layers; ++il) {
        triattention_copy_layer_params(&cal->layers[il], &model->layers[il]);
    }
    triattention_update_summary_from_layers(cal, cal->layers, cal->num_layers);

    snprintf(cal->model_name, sizeof(cal->model_name), "%s", "runtime-fallback");

    uint32_t sampled = 0;
    for (uint32_t layer = 0; layer < model->num_layers; ++layer) {
        const triattention_layer_params & lp = model->layers[layer];
        if (!keep_layer[layer] || lp.num_kv_heads == 0 || lp.num_kv_groups == 0 || lp.kv_source_layer == TRIATTENTION_LAYER_NONE) {
            continue;
        }
        sampled += lp.num_kv_heads;
    }

    cal->n_sampled = sampled;
    cal->sampled_layer = new uint32_t[sampled];
    cal->sampled_head  = new uint32_t[sampled];
    cal->head_stats    = new triattention_head_stats[sampled];
    memset(cal->head_stats, 0, sizeof(triattention_head_stats) * sampled);

    uint32_t idx = 0;
    for (uint32_t layer = 0; layer < model->num_layers; ++layer) {
        const triattention_layer_params & lp = model->layers[layer];
        if (!keep_layer[layer] || lp.num_kv_heads == 0 || lp.num_kv_groups == 0 || lp.kv_source_layer == TRIATTENTION_LAYER_NONE) {
            continue;
        }
        for (uint32_t kv_head = 0; kv_head < lp.num_kv_heads; ++kv_head) {
            cal->sampled_layer[idx] = layer;
            cal->sampled_head[idx]  = kv_head * lp.num_kv_groups;
            ++idx;
        }
    }

    if (!cal->heterogeneous_layout && cal->num_layers > 0 && cal->layers[0].omega && cal->layers[0].freq_scale_sq) {
        cal->omega = new float[cal->layers[0].freq_count];
        cal->freq_scale_sq = new float[cal->layers[0].freq_count];
        memcpy(cal->omega, cal->layers[0].omega, sizeof(float) * cal->layers[0].freq_count);
        memcpy(cal->freq_scale_sq, cal->layers[0].freq_scale_sq, sizeof(float) * cal->layers[0].freq_count);
    }

    return cal;
}
