#include "llama-triattention-file.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static float triattention_rope_yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2.0f - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static float triattention_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    static const float kPi = 3.14159265358979323846f;
    return n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * kPi)) / (2.0f * logf(base));
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

bool triattention_build_rope_arrays(
    float * omega,
    float * freq_scale_sq,
    uint32_t freq_count,
    const triattention_rope_params * params) {
    if (!omega || !freq_scale_sq || !params || params->n_dims == 0 || (params->n_dims % 2) != 0) {
        return false;
    }

    const float freq_base  = params->freq_base > 0.0f ? params->freq_base : 10000.0f;
    const float freq_scale = params->freq_scale > 0.0f ? params->freq_scale : 1.0f;
    const float ext_factor = params->ext_factor > 0.0f ? params->ext_factor : 0.0f;
    const float attn_factor = params->attn_factor > 0.0f ? params->attn_factor : 1.0f;
    const float beta_fast = params->beta_fast > 0.0f ? params->beta_fast : 32.0f;
    const float beta_slow = params->beta_slow > 0.0f ? params->beta_slow : 1.0f;
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
    delete cal;
}

triattention_calibration * triattention_calibration_load(const char * path, bool verbose) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[TriAttention] ERROR: cannot open calibration file: %s\n", path);
        return nullptr;
    }

    uint32_t magic = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != TRIATTENTION_MAGIC) {
        fprintf(stderr, "[TriAttention] ERROR: invalid magic in %s (got 0x%08x, expected 0x%08x)\n",
                path, magic, TRIATTENTION_MAGIC);
        fclose(f);
        return nullptr;
    }

    uint32_t version = 0;
    if (fread(&version, sizeof(version), 1, f) != 1 || (version != 1u && version != TRIATTENTION_VERSION)) {
        fprintf(stderr, "[TriAttention] ERROR: unsupported version %u in %s\n", version, path);
        fclose(f);
        return nullptr;
    }

    auto * cal = new triattention_calibration();
    memset(cal, 0, sizeof(*cal));
    cal->version = version;

    bool ok = true;
    ok = ok && fread(&cal->head_dim,       sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_layers,     sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_attn_heads, sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->num_kv_heads,   sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->rope_theta,     sizeof(double),   1, f) == 1;
    ok = ok && fread(&cal->rope_style,     sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->n_sampled,      sizeof(uint32_t), 1, f) == 1;
    ok = ok && fread(&cal->freq_count,     sizeof(uint32_t), 1, f) == 1;

    if (!ok) {
        fprintf(stderr, "[TriAttention] ERROR: truncated header in %s\n", path);
        triattention_calibration_free(cal);
        fclose(f);
        return nullptr;
    }

    uint32_t name_len = 0;
    if (fread(&name_len, sizeof(name_len), 1, f) != 1 || name_len == 0 || name_len > sizeof(cal->model_name)) {
        fprintf(stderr, "[TriAttention] ERROR: invalid model name length %u in %s\n", name_len, path);
        triattention_calibration_free(cal);
        fclose(f);
        return nullptr;
    }

    if (fread(cal->model_name, 1, name_len, f) != name_len) {
        fprintf(stderr, "[TriAttention] ERROR: truncated model name in %s\n", path);
        triattention_calibration_free(cal);
        fclose(f);
        return nullptr;
    }
    cal->model_name[sizeof(cal->model_name) - 1] = '\0';

    if (version >= TRIATTENTION_VERSION) {
        cal->omega = new float[cal->freq_count];
        cal->freq_scale_sq = new float[cal->freq_count];

        ok = true;
        ok = ok && fread(cal->omega,         sizeof(float), cal->freq_count, f) == cal->freq_count;
        ok = ok && fread(cal->freq_scale_sq, sizeof(float), cal->freq_count, f) == cal->freq_count;
        if (!ok) {
            fprintf(stderr, "[TriAttention] ERROR: truncated v2 rope arrays in %s\n", path);
            triattention_calibration_free(cal);
            fclose(f);
            return nullptr;
        }
    }

    if (cal->freq_count != cal->head_dim / 2) {
        fprintf(stderr, "[TriAttention] ERROR: freq_count (%u) != head_dim/2 (%u) in %s\n",
                cal->freq_count, cal->head_dim / 2, path);
        triattention_calibration_free(cal);
        fclose(f);
        return nullptr;
    }

    if (cal->num_attn_heads == 0 || cal->num_kv_heads == 0 || cal->num_layers == 0 ||
        cal->num_attn_heads % cal->num_kv_heads != 0) {
        fprintf(stderr, "[TriAttention] ERROR: invalid model dimensions in %s\n", path);
        triattention_calibration_free(cal);
        fclose(f);
        return nullptr;
    }

    cal->num_kv_groups = cal->num_attn_heads / cal->num_kv_heads;
    cal->sampled_layer = new uint32_t[cal->n_sampled];
    cal->sampled_head  = new uint32_t[cal->n_sampled];
    cal->head_stats    = new triattention_head_stats[cal->n_sampled];
    memset(cal->head_stats, 0, sizeof(triattention_head_stats) * cal->n_sampled);

    for (uint32_t h = 0; h < cal->n_sampled; ++h) {
        auto & hs = cal->head_stats[h];
        ok = true;
        ok = ok && fread(&cal->sampled_layer[h], sizeof(uint32_t), 1, f) == 1;
        ok = ok && fread(&cal->sampled_head[h],  sizeof(uint32_t), 1, f) == 1;
        if (!ok) {
            fprintf(stderr, "[TriAttention] ERROR: truncated head entry %u in %s\n", h, path);
            triattention_calibration_free(cal);
            fclose(f);
            return nullptr;
        }

        hs.q_mean_real = new float[cal->freq_count];
        hs.q_mean_imag = new float[cal->freq_count];
        hs.q_abs_mean  = new float[cal->freq_count];
        hs.r_f         = new float[cal->freq_count];

        ok = true;
        ok = ok && fread(hs.q_mean_real, sizeof(float), cal->freq_count, f) == cal->freq_count;
        ok = ok && fread(hs.q_mean_imag, sizeof(float), cal->freq_count, f) == cal->freq_count;
        ok = ok && fread(hs.q_abs_mean,  sizeof(float), cal->freq_count, f) == cal->freq_count;
        ok = ok && fread(hs.r_f,         sizeof(float), cal->freq_count, f) == cal->freq_count;
        if (!ok) {
            fprintf(stderr, "[TriAttention] ERROR: truncated stats for head %u in %s\n", h, path);
            triattention_calibration_free(cal);
            fclose(f);
            return nullptr;
        }
    }

    fclose(f);

    if (verbose) {
        fprintf(stderr, "[TriAttention] Loaded calibration: model=%s, version=%u, layers=%u, attn_heads=%u, kv_heads=%u, head_dim=%u, sampled=%u\n",
                cal->model_name, cal->version, cal->num_layers, cal->num_attn_heads,
                cal->num_kv_heads, cal->head_dim, cal->n_sampled);
    }

    return cal;
}

bool triattention_calibration_save(const char * path, const triattention_calibration * cal) {
    if (!path || !cal) {
        return false;
    }

    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[TriAttention] ERROR: cannot write calibration file: %s\n", path);
        return false;
    }

    const uint32_t magic = TRIATTENTION_MAGIC;
    const uint32_t version = cal->version == 0 ? TRIATTENTION_VERSION : cal->version;
    const uint32_t name_len = (uint32_t) strnlen(cal->model_name, sizeof(cal->model_name) - 1) + 1;

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

    if (version >= TRIATTENTION_VERSION) {
        ok = ok && cal->omega != nullptr && cal->freq_scale_sq != nullptr;
        ok = ok && fwrite(cal->omega,         sizeof(float), cal->freq_count, f) == cal->freq_count;
        ok = ok && fwrite(cal->freq_scale_sq, sizeof(float), cal->freq_count, f) == cal->freq_count;
    }

    for (uint32_t h = 0; ok && h < cal->n_sampled; ++h) {
        const auto & hs = cal->head_stats[h];
        ok = ok && hs.q_mean_real != nullptr && hs.q_mean_imag != nullptr && hs.q_abs_mean != nullptr && hs.r_f != nullptr;
        ok = ok && fwrite(&cal->sampled_layer[h], sizeof(uint32_t), 1, f) == 1;
        ok = ok && fwrite(&cal->sampled_head[h],  sizeof(uint32_t), 1, f) == 1;
        ok = ok && fwrite(hs.q_mean_real, sizeof(float), cal->freq_count, f) == cal->freq_count;
        ok = ok && fwrite(hs.q_mean_imag, sizeof(float), cal->freq_count, f) == cal->freq_count;
        ok = ok && fwrite(hs.q_abs_mean,  sizeof(float), cal->freq_count, f) == cal->freq_count;
        ok = ok && fwrite(hs.r_f,         sizeof(float), cal->freq_count, f) == cal->freq_count;
    }

    fclose(f);

    if (!ok) {
        fprintf(stderr, "[TriAttention] ERROR: failed while writing calibration file: %s\n", path);
        return false;
    }

    return true;
}

bool triattention_calibration_validate(
    const triattention_calibration * cal,
    const triattention_model_params * model,
    bool warn_rope_theta) {
    if (!cal || !model) {
        return false;
    }

    if (cal->head_dim != model->head_dim) {
        fprintf(stderr, "[TriAttention] ERROR: head_dim mismatch (calibration=%u, model=%u)\n",
                cal->head_dim, model->head_dim);
        return false;
    }
    if (cal->num_layers != model->num_layers) {
        fprintf(stderr, "[TriAttention] ERROR: layer count mismatch (calibration=%u, model=%u)\n",
                cal->num_layers, model->num_layers);
        return false;
    }
    if (cal->num_attn_heads != model->num_attn_heads) {
        fprintf(stderr, "[TriAttention] ERROR: attention head mismatch (calibration=%u, model=%u)\n",
                cal->num_attn_heads, model->num_attn_heads);
        return false;
    }
    if (cal->num_kv_heads != model->num_kv_heads) {
        fprintf(stderr, "[TriAttention] ERROR: KV head mismatch (calibration=%u, model=%u)\n",
                cal->num_kv_heads, model->num_kv_heads);
        return false;
    }
    if (cal->rope_style != model->rope_style) {
        fprintf(stderr, "[TriAttention] ERROR: rope style mismatch (calibration=%u, model=%u)\n",
                cal->rope_style, model->rope_style);
        return false;
    }

    if (warn_rope_theta && fabs(cal->rope_theta - model->rope_theta) / fmax(cal->rope_theta, 1.0) > 0.01) {
        fprintf(stderr, "[TriAttention] WARNING: rope_theta mismatch (calibration=%.1f, model=%.1f)\n",
                cal->rope_theta, model->rope_theta);
    }

    return true;
}

triattention_calibration * triattention_calibration_create_fallback(
    const triattention_model_params * model) {
    if (!model || model->num_layers == 0 || model->num_kv_heads == 0 || model->head_dim == 0) {
        return nullptr;
    }

    auto * cal = new triattention_calibration();
    memset(cal, 0, sizeof(*cal));

    cal->version        = TRIATTENTION_VERSION;
    cal->head_dim       = model->head_dim;
    cal->num_layers     = model->num_layers;
    cal->num_attn_heads = model->num_attn_heads;
    cal->num_kv_heads   = model->num_kv_heads;
    cal->num_kv_groups  = model->num_attn_heads / model->num_kv_heads;
    cal->rope_theta     = model->rope_theta;
    cal->rope_style     = model->rope_style;
    cal->freq_count     = model->head_dim / 2;
    cal->n_sampled      = model->num_layers * model->num_kv_heads;

    snprintf(cal->model_name, sizeof(cal->model_name), "%s", "runtime-fallback");

    cal->sampled_layer = new uint32_t[cal->n_sampled];
    cal->sampled_head  = new uint32_t[cal->n_sampled];
    cal->head_stats    = new triattention_head_stats[cal->n_sampled];
    memset(cal->head_stats, 0, sizeof(triattention_head_stats) * cal->n_sampled);

    uint32_t idx = 0;
    for (uint32_t layer = 0; layer < model->num_layers; ++layer) {
        for (uint32_t kv_head = 0; kv_head < model->num_kv_heads; ++kv_head) {
            cal->sampled_layer[idx] = layer;
            cal->sampled_head[idx] = kv_head * cal->num_kv_groups;
            ++idx;
        }
    }

    return cal;
}
