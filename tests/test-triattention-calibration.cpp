#include "ggml.h"
#include "llama-triattention-calibration.h"
#include "testing.h"

#include <cmath>
#include <cstring>

static bool approx_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
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

        triattention_calibration_builder builder("synthetic", 1, 1, 1);
        std::string error;
        t.assert_true(builder.accumulate_query_tensor(tensor, tensor->data, 0, 0, rope_params, &error));

        triattention_calibration * cal = builder.finalize(&error);
        t.assert_true(error, cal != nullptr);
        t.assert_equal((uint32_t) 1, cal->n_sampled);
        t.assert_equal((uint32_t) 2, cal->freq_count);

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
        ggml_free(ctx);
    });

    return t.summary();
}
