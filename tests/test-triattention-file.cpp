#include "llama-triattention-file.h"
#include "testing.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

static bool approx_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

static std::filesystem::path temp_path(const char * name) {
    return std::filesystem::temp_directory_path() /
        (std::string(name) + "-" + std::to_string(std::rand()) + ".triattention");
}

static triattention_calibration * make_v2_calibration() {
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
        triattention_calibration * cal = make_v2_calibration();

        t.assert_true(triattention_calibration_save(path.string().c_str(), cal));

        triattention_calibration * loaded = triattention_calibration_load(path.string().c_str(), false);
        t.assert_true(loaded != nullptr);
        t.assert_equal((uint32_t) TRIATTENTION_VERSION, loaded->version);
        t.assert_equal((uint32_t) 2, loaded->n_sampled);
        t.assert_true(loaded->omega != nullptr);
        t.assert_true(loaded->freq_scale_sq != nullptr);
        t.assert_true(approx_equal(loaded->omega[1], 0.5f));
        t.assert_true(approx_equal(loaded->freq_scale_sq[0], 2.0f));
        t.assert_true(approx_equal(loaded->head_stats[1].q_mean_imag[1], 5.0f));
        t.assert_true(approx_equal(loaded->head_stats[1].r_f[0], 0.2f));

        triattention_calibration_free(loaded);
        triattention_calibration_free(cal);
        std::filesystem::remove(path);
    });

    t.test("v1 calibration loads and validates", [](testing & t) {
        const std::filesystem::path path = temp_path("triattention-v1");
        t.assert_true(write_v1_fixture(path));

        triattention_calibration * cal = triattention_calibration_load(path.string().c_str(), false);
        t.assert_true(cal != nullptr);
        t.assert_equal((uint32_t) 1, cal->version);
        t.assert_true(cal->omega == nullptr);
        t.assert_true(cal->freq_scale_sq == nullptr);

        triattention_model_params model = {};
        model.head_dim = 4;
        model.num_layers = 1;
        model.num_attn_heads = 2;
        model.num_kv_heads = 1;
        model.rope_style = 0;
        model.rope_theta = 10000.0;
        t.assert_true(triattention_calibration_validate(cal, &model, true));

        triattention_calibration_free(cal);
        std::filesystem::remove(path);
    });

    t.test("validation rejects head mismatch", [](testing & t) {
        triattention_calibration * cal = make_v2_calibration();
        triattention_model_params model = {};
        model.head_dim = 8;
        model.num_layers = cal->num_layers;
        model.num_attn_heads = cal->num_attn_heads;
        model.num_kv_heads = cal->num_kv_heads;
        model.rope_style = cal->rope_style;
        model.rope_theta = cal->rope_theta;

        t.assert_true(!triattention_calibration_validate(cal, &model, false));
        triattention_calibration_free(cal);
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
