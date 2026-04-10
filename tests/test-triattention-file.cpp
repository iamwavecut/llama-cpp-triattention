#include "llama-triattention-file.h"
#include "testing.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <string>

static bool approx_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

static std::filesystem::path temp_path(const char * name) {
    return std::filesystem::temp_directory_path() /
        (std::string(name) + "-" + std::to_string(std::rand()) + ".triattention");
}

static void populate_uniform_layer(triattention_layer_params * layer, uint32_t kv_source_layer) {
    std::memset(layer, 0, sizeof(*layer));
    layer->head_dim = 4;
    layer->rope_dim = 4;
    layer->rope_offset = 0;
    layer->num_attn_heads = 4;
    layer->num_kv_heads = 2;
    layer->num_kv_groups = 2;
    layer->kv_source_layer = kv_source_layer;
    layer->rope_style = 0;
    layer->freq_count = 2;
    layer->n_ctx_orig = 128;
    layer->rope_theta = 10000.0;
    layer->rope_freq_scale = 1.0f;
    layer->rope_ext_factor = 0.0f;
    layer->rope_attn_factor = 1.0f;
    layer->rope_beta_fast = 32.0f;
    layer->rope_beta_slow = 1.0f;
    layer->omega = new float[2] { 1.0f, 0.5f };
    layer->freq_scale_sq = new float[2] { 2.0f, 3.0f };
}

static triattention_model_params make_uniform_model_params(uint32_t num_layers, uint32_t head_dim = 4) {
    triattention_model_params model = {};
    model.num_layers = num_layers;
    model.head_dim = head_dim;
    model.rope_dim = head_dim;
    model.num_attn_heads = 4;
    model.num_kv_heads = 2;
    model.rope_style = 0;
    model.n_ctx_orig = 128;
    model.rope_theta = 10000.0;
    model.rope_freq_scale = 1.0f;
    model.rope_ext_factor = 0.0f;
    model.rope_attn_factor = 1.0f;
    model.rope_beta_fast = 32.0f;
    model.rope_beta_slow = 1.0f;
    model.max_head_dim = head_dim;
    model.max_rope_dim = head_dim;
    model.max_freq_count = head_dim / 2;
    model.layers = new triattention_layer_params[num_layers];
    for (uint32_t il = 0; il < num_layers; ++il) {
        populate_uniform_layer(&model.layers[il], il);
        model.layers[il].head_dim = head_dim;
        model.layers[il].rope_dim = head_dim;
        model.layers[il].freq_count = head_dim / 2;
        delete[] model.layers[il].omega;
        delete[] model.layers[il].freq_scale_sq;
        model.layers[il].omega = new float[head_dim / 2];
        model.layers[il].freq_scale_sq = new float[head_dim / 2];
        for (uint32_t f = 0; f < head_dim / 2; ++f) {
            model.layers[il].omega[f] = 1.0f / (float) (f + 1);
            model.layers[il].freq_scale_sq[f] = 1.0f + (float) f;
        }
    }
    return model;
}

static triattention_calibration * make_v3_calibration() {
    auto * cal = new triattention_calibration();
    std::memset(cal, 0, sizeof(*cal));

    cal->version        = TRIATTENTION_VERSION;
    cal->head_dim       = 4;
    cal->num_layers     = 2;
    cal->num_attn_heads = 4;
    cal->num_kv_heads   = 2;
    cal->num_kv_groups  = 2;
    cal->rope_theta     = 10000.0;
    cal->rope_style     = 0;
    cal->freq_count     = 2;
    cal->n_sampled      = 2;
    std::snprintf(cal->model_name, sizeof(cal->model_name), "%s", "synthetic-v2");

    cal->omega = new float[2] { 1.0f, 0.5f };
    cal->freq_scale_sq = new float[2] { 2.0f, 3.0f };
    cal->layers = new triattention_layer_params[2];
    populate_uniform_layer(&cal->layers[0], 0);
    populate_uniform_layer(&cal->layers[1], 1);
    cal->max_head_dim = 4;
    cal->max_rope_dim = 4;
    cal->max_freq_count = 2;
    cal->sampled_layer = new uint32_t[2] { 0, 1 };
    cal->sampled_head  = new uint32_t[2] { 0, 2 };
    cal->head_stats    = new triattention_head_stats[2];
    std::memset(cal->head_stats, 0, sizeof(triattention_head_stats) * 2);

    for (uint32_t i = 0; i < cal->n_sampled; ++i) {
        auto & hs = cal->head_stats[i];
        hs.q_mean_real = new float[2] { 1.0f + i, 2.0f + i };
        hs.q_mean_imag = new float[2] { 3.0f + i, 4.0f + i };
        hs.q_abs_mean  = new float[2] { 5.0f + i, 6.0f + i };
        hs.r_f         = new float[2] { 0.1f * (i + 1), 0.2f * (i + 1) };
    }

    return cal;
}

static bool write_v1_fixture(const std::filesystem::path & path) {
    FILE * f = std::fopen(path.string().c_str(), "wb");
    if (!f) {
        return false;
    }

    const uint32_t magic = TRIATTENTION_MAGIC;
    const uint32_t version = 1;
    const uint32_t head_dim = 4;
    const uint32_t num_layers = 1;
    const uint32_t num_attn_heads = 2;
    const uint32_t num_kv_heads = 1;
    const double rope_theta = 10000.0;
    const uint32_t rope_style = 0;
    const uint32_t n_sampled = 1;
    const uint32_t freq_count = 2;
    const char name[] = "synthetic-v1";
    const uint32_t name_len = (uint32_t) sizeof(name);
    const uint32_t layer = 0;
    const uint32_t head = 0;
    const float q_mean_real[2] = { 1.0f, 2.0f };
    const float q_mean_imag[2] = { 3.0f, 4.0f };
    const float q_abs_mean[2]  = { 5.0f, 6.0f };
    const float r_f[2]         = { 0.25f, 0.50f };

    bool ok = true;
    ok = ok && std::fwrite(&magic, sizeof(magic), 1, f) == 1;
    ok = ok && std::fwrite(&version, sizeof(version), 1, f) == 1;
    ok = ok && std::fwrite(&head_dim, sizeof(head_dim), 1, f) == 1;
    ok = ok && std::fwrite(&num_layers, sizeof(num_layers), 1, f) == 1;
    ok = ok && std::fwrite(&num_attn_heads, sizeof(num_attn_heads), 1, f) == 1;
    ok = ok && std::fwrite(&num_kv_heads, sizeof(num_kv_heads), 1, f) == 1;
    ok = ok && std::fwrite(&rope_theta, sizeof(rope_theta), 1, f) == 1;
    ok = ok && std::fwrite(&rope_style, sizeof(rope_style), 1, f) == 1;
    ok = ok && std::fwrite(&n_sampled, sizeof(n_sampled), 1, f) == 1;
    ok = ok && std::fwrite(&freq_count, sizeof(freq_count), 1, f) == 1;
    ok = ok && std::fwrite(&name_len, sizeof(name_len), 1, f) == 1;
    ok = ok && std::fwrite(name, 1, name_len, f) == name_len;
    ok = ok && std::fwrite(&layer, sizeof(layer), 1, f) == 1;
    ok = ok && std::fwrite(&head, sizeof(head), 1, f) == 1;
    ok = ok && std::fwrite(q_mean_real, sizeof(float), 2, f) == 2;
    ok = ok && std::fwrite(q_mean_imag, sizeof(float), 2, f) == 2;
    ok = ok && std::fwrite(q_abs_mean, sizeof(float), 2, f) == 2;
    ok = ok && std::fwrite(r_f, sizeof(float), 2, f) == 2;

    std::fclose(f);
    return ok;
}

int main() {
    testing t;

    t.test("v2 calibration roundtrip", [](testing & t) {
        const std::filesystem::path path = temp_path("triattention-v2");
        triattention_calibration * cal = make_v3_calibration();

        t.assert_true(triattention_calibration_save(path.string().c_str(), cal));

        triattention_calibration * loaded = triattention_calibration_load(path.string().c_str(), false);
        t.assert_true(loaded != nullptr);
        t.assert_equal((uint32_t) TRIATTENTION_VERSION, loaded->version);
        t.assert_equal((uint32_t) 2, loaded->n_sampled);
        t.assert_true(loaded->layers != nullptr);
        t.assert_true(loaded->omega != nullptr);
        t.assert_true(loaded->freq_scale_sq != nullptr);
        t.assert_true(approx_equal(loaded->omega[1], 0.5f));
        t.assert_true(approx_equal(loaded->freq_scale_sq[0], 2.0f));
        t.assert_equal((uint32_t) 1, loaded->layers[1].kv_source_layer);
        t.assert_true(approx_equal(loaded->head_stats[1].q_mean_imag[1], 5.0f));
        t.assert_true(approx_equal(loaded->head_stats[1].r_f[0], 0.2f));

        triattention_calibration_free(loaded);
        triattention_calibration_free(cal);
        std::filesystem::remove(path);
    });

    t.test("buffer roundtrip preserves calibration", [](testing & t) {
        triattention_calibration * cal = make_v3_calibration();
        std::vector<uint8_t> buffer;

        t.assert_true(triattention_calibration_save_to_buffer(cal, buffer));
        t.assert_true(!buffer.empty());

        triattention_calibration * loaded = triattention_calibration_load_from_buffer(
            buffer.data(), buffer.size(), false, "buffer-roundtrip");
        t.assert_true(loaded != nullptr);
        t.assert_equal((uint32_t) TRIATTENTION_VERSION, loaded->version);
        t.assert_equal((uint32_t) 2, loaded->n_sampled);
        t.assert_true(approx_equal(loaded->layers[0].omega[0], 1.0f));
        t.assert_true(approx_equal(loaded->layers[1].freq_scale_sq[1], 3.0f));
        t.assert_true(approx_equal(loaded->head_stats[0].q_abs_mean[1], 6.0f));

        triattention_calibration_free(loaded);
        triattention_calibration_free(cal);
    });

    t.test("v1 calibration loads and validates", [](testing & t) {
        const std::filesystem::path path = temp_path("triattention-v1");
        t.assert_true(write_v1_fixture(path));

        triattention_calibration * cal = triattention_calibration_load(path.string().c_str(), false);
        t.assert_true(cal != nullptr);
        t.assert_equal((uint32_t) 1, cal->version);
        t.assert_true(cal->omega == nullptr);
        t.assert_true(cal->freq_scale_sq == nullptr);

        triattention_model_params model = make_uniform_model_params(1);
        model.num_attn_heads = 2;
        model.num_kv_heads = 1;
        model.layers[0].num_attn_heads = 2;
        model.layers[0].num_kv_heads = 1;
        model.layers[0].num_kv_groups = 2;
        t.assert_true(triattention_calibration_validate(cal, &model, true));

        triattention_calibration_free(cal);
        triattention_model_params_clear(&model);
        std::filesystem::remove(path);
    });

    t.test("validation rejects head mismatch", [](testing & t) {
        triattention_calibration * cal = make_v3_calibration();
        triattention_model_params model = make_uniform_model_params(cal->num_layers);
        model.head_dim = 8;
        model.rope_dim = 8;
        model.max_head_dim = 8;
        model.max_rope_dim = 8;
        model.max_freq_count = 4;
        model.layers[0].head_dim = 8;
        model.layers[0].rope_dim = 8;
        model.layers[0].freq_count = 4;

        t.assert_true(!triattention_calibration_validate(cal, &model, false));
        triattention_calibration_free(cal);
        triattention_model_params_clear(&model);
    });

    t.test("truncated file is rejected", [](testing & t) {
        const std::filesystem::path path = temp_path("triattention-bad");
        FILE * f = std::fopen(path.string().c_str(), "wb");
        t.assert_true(f != nullptr);
        const uint32_t magic = TRIATTENTION_MAGIC;
        const uint32_t version = TRIATTENTION_VERSION;
        std::fwrite(&magic, sizeof(magic), 1, f);
        std::fwrite(&version, sizeof(version), 1, f);
        std::fclose(f);

        triattention_calibration * cal = triattention_calibration_load(path.string().c_str(), false);
        t.assert_true(cal == nullptr);
        std::filesystem::remove(path);
    });

    return t.summary();
}
