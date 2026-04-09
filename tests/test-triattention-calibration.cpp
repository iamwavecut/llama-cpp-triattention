#include "ggml.h"
#include "llama-triattention-calibration.h"
#include "testing.h"

#include <cmath>
#include <cstring>

static bool approx_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

static triattention_model_params make_synthetic_model() {
    triattention_model_params model = {};
    model.num_layers = 1;
    model.head_dim = 4;
    model.rope_dim = 4;
    model.num_attn_heads = 1;
    model.num_kv_heads = 1;
    model.rope_style = 0;
    model.n_ctx_orig = 128;
    model.rope_theta = 10000.0;
    model.rope_freq_scale = 1.0f;
    model.rope_ext_factor = 0.0f;
    model.rope_attn_factor = 1.0f;
    model.rope_beta_fast = 32.0f;
    model.rope_beta_slow = 1.0f;
    model.max_head_dim = 4;
    model.max_rope_dim = 4;
    model.max_freq_count = 2;
    model.layers = new triattention_layer_params[1];
    std::memset(model.layers, 0, sizeof(triattention_layer_params));

    triattention_layer_params & layer = model.layers[0];
    layer.head_dim = 4;
    layer.rope_dim = 4;
    layer.rope_offset = 0;
    layer.num_attn_heads = 1;
    layer.num_kv_heads = 1;
    layer.num_kv_groups = 1;
    layer.kv_source_layer = 0;
    layer.rope_style = 0;
    layer.freq_count = 2;
    layer.n_ctx_orig = 128;
    layer.rope_theta = 10000.0;
    layer.rope_freq_scale = 1.0f;
    layer.rope_ext_factor = 0.0f;
    layer.rope_attn_factor = 1.0f;
    layer.rope_beta_fast = 32.0f;
    layer.rope_beta_slow = 1.0f;
    layer.omega = new float[2] { 1.0f, 0.1f };
    layer.freq_scale_sq = new float[2] { 1.0f, 1.0f };

    return model;
}

int main() {
    testing t;

    t.test("query rope name matching", [](testing & t) {
        t.assert_true(triattention_is_query_rope_name("Qcur-0"));
        t.assert_true(triattention_is_query_rope_name("Qcur-0 (reshaped)"));
        t.assert_true(triattention_is_query_rope_name("q_pe-12"));
        t.assert_true(!triattention_is_query_rope_name("Kcur-0"));
    });

    t.test("layer index parsing", [](testing & t) {
        uint32_t layer = 0;
        t.assert_true(triattention_parse_layer_index("Qcur-17", &layer));
        t.assert_equal((uint32_t) 17, layer);

        t.assert_true(triattention_parse_layer_index("Qcur-3 (reshaped)", &layer));
        t.assert_equal((uint32_t) 3, layer);

        t.assert_true(!triattention_parse_layer_index("Qcur-final", &layer));
    });

    t.test("streaming accumulator computes q statistics", [](testing & t) {
        ggml_init_params params = {
            /*.mem_size   =*/ 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ggml_context * ctx = ggml_init(params);
        t.assert_true(ctx != nullptr);

        ggml_tensor * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 1, 2);
        std::strncpy(tensor->name, "Qcur-0", sizeof(tensor->name) - 1);

        auto * data = static_cast<float *>(tensor->data);
        data[0] = 1.0f;
        data[1] = 3.0f;
        data[2] = 2.0f;
        data[3] = 4.0f;
        data[4] = 5.0f;
        data[5] = 7.0f;
        data[6] = 6.0f;
        data[7] = 8.0f;

        triattention_rope_params rope_params = {
            /*.n_dims            =*/ 4,
            /*.n_ctx_orig        =*/ 128,
            /*.freq_base         =*/ 10000.0f,
            /*.freq_scale        =*/ 1.0f,
            /*.ext_factor        =*/ 0.0f,
            /*.attn_factor       =*/ 1.0f,
            /*.beta_fast         =*/ 32.0f,
            /*.beta_slow         =*/ 1.0f,
            /*.freq_factors      =*/ nullptr,
            /*.freq_factor_count =*/ 0,
        };

        triattention_model_params model = make_synthetic_model();
        triattention_calibration_builder builder("synthetic", &model);
        std::string error;
        t.assert_true(builder.accumulate_query_tensor(tensor, tensor->data, 0, 0, rope_params, &error));

        triattention_calibration * cal = builder.finalize(&error);
        t.assert_true(error, cal != nullptr);
        t.assert_equal((uint32_t) 1, cal->n_sampled);
        t.assert_equal((uint32_t) 2, cal->freq_count);
        t.assert_equal((uint32_t) 4, cal->layers[0].head_dim);
        t.assert_equal((uint32_t) 4, cal->layers[0].rope_dim);
        t.assert_true(cal->layers[0].omega != nullptr);
        t.assert_true(cal->layers[0].freq_scale_sq != nullptr);

        const triattention_head_stats & hs = cal->head_stats[0];
        const float abs0 = 0.5f * (std::sqrt(5.0f) + std::sqrt(61.0f));
        const float abs1 = 0.5f * (5.0f + std::sqrt(113.0f));
        const float rf0 = std::sqrt(3.0f * 3.0f + 4.0f * 4.0f) / abs0;
        const float rf1 = std::sqrt(5.0f * 5.0f + 6.0f * 6.0f) / abs1;

        t.assert_true(approx_equal(hs.q_mean_real[0], 3.0f));
        t.assert_true(approx_equal(hs.q_mean_real[1], 5.0f));
        t.assert_true(approx_equal(hs.q_mean_imag[0], 4.0f));
        t.assert_true(approx_equal(hs.q_mean_imag[1], 6.0f));
        t.assert_true(approx_equal(hs.q_abs_mean[0], abs0));
        t.assert_true(approx_equal(hs.q_abs_mean[1], abs1));
        t.assert_true(approx_equal(hs.r_f[0], rf0));
        t.assert_true(approx_equal(hs.r_f[1], rf1));

        triattention_calibration_free(cal);
        triattention_model_params_clear(&model);
        ggml_free(ctx);
    });

    return t.summary();
}
