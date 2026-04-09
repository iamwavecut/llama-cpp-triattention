#include "llama-triattention-file.h"
#include "llama-triattention.h"
#include "testing.h"

#include <cmath>
#include <cstring>

static bool approx_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

static triattention_model_params make_fallback_model() {
    triattention_model_params model = {};
    model.num_layers = 2;
    model.head_dim = 8;
    model.rope_dim = 8;
    model.num_attn_heads = 8;
    model.num_kv_heads = 2;
    model.rope_style = 0;
    model.n_ctx_orig = 128;
    model.rope_theta = 10000.0;
    model.rope_freq_scale = 1.0f;
    model.rope_ext_factor = 0.0f;
    model.rope_attn_factor = 1.0f;
    model.rope_beta_fast = 32.0f;
    model.rope_beta_slow = 1.0f;
    model.max_head_dim = 8;
    model.max_rope_dim = 8;
    model.max_freq_count = 4;
    model.layers = new triattention_layer_params[2];
    std::memset(model.layers, 0, sizeof(triattention_layer_params) * 2);

    for (uint32_t il = 0; il < model.num_layers; ++il) {
        triattention_layer_params & layer = model.layers[il];
        layer.head_dim = 8;
        layer.rope_dim = 8;
        layer.rope_offset = 0;
        layer.num_attn_heads = 8;
        layer.num_kv_heads = 2;
        layer.num_kv_groups = 4;
        layer.kv_source_layer = il;
        layer.rope_style = 0;
        layer.freq_count = 4;
        layer.n_ctx_orig = 128;
        layer.rope_theta = 10000.0;
        layer.rope_freq_scale = 1.0f;
        layer.rope_ext_factor = 0.0f;
        layer.rope_attn_factor = 1.0f;
        layer.rope_beta_fast = 32.0f;
        layer.rope_beta_slow = 1.0f;
        layer.omega = new float[4] { 1.0f, 0.5f, 0.25f, 0.125f };
        layer.freq_scale_sq = new float[4] { 1.0f, 1.0f, 1.0f, 1.0f };
    }

    return model;
}

int main() {
    testing t;

    t.test("norm scorer prefers larger key magnitude", [](testing & t) {
        const float pre_rope_k[] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            2.0f, 0.0f, 0.0f, 0.0f,
        };
        const float freq_scale_sq[] = { 1.0f, 1.0f };
        float scores[2] = { 0.0f, 0.0f };

        triattention_score_keys_norm(scores, pre_rope_k, freq_scale_sq, 2, 4, 2);
        t.assert_true(scores[1] > scores[0]);
    });

    t.test("fallback recency helper orders newer tokens higher", [](testing & t) {
        const int32_t positions[] = { 10, 20, 30 };
        float recency[3] = { 0.0f, 0.0f, 0.0f };

        triattention_build_recency_scores(recency, positions, 3);
        t.assert_true(approx_equal(recency[0], 0.0f));
        t.assert_true(approx_equal(recency[1], 0.5f));
        t.assert_true(approx_equal(recency[2], 1.0f));
    });

    t.test("fallback blend respects lambda extremes", [](testing & t) {
        const float recency[] = { 0.0f, 0.5f, 1.0f };

        float scores0[] = { 10.0f, 10.0f, 30.0f };
        triattention_blend_fallback_scores(scores0, recency, 3, 0.0f);
        t.assert_true(approx_equal(scores0[0], 0.0f));
        t.assert_true(approx_equal(scores0[1], 0.0f));
        t.assert_true(approx_equal(scores0[2], 1.0f));

        float scores1[] = { 10.0f, 10.0f, 30.0f };
        triattention_blend_fallback_scores(scores1, recency, 3, 1.0f);
        t.assert_true(approx_equal(scores1[0], 0.0f));
        t.assert_true(approx_equal(scores1[1], 0.5f));
        t.assert_true(approx_equal(scores1[2], 1.0f));
    });

    t.test("fallback calibration samples representative kv heads", [](testing & t) {
        triattention_model_params model = make_fallback_model();

        triattention_calibration * cal = triattention_calibration_create_fallback(&model);
        t.assert_true(cal != nullptr);
        t.assert_equal((uint32_t) 4, cal->n_sampled);
        t.assert_equal((uint32_t) 0, cal->sampled_layer[0]);
        t.assert_equal((uint32_t) 0, cal->sampled_head[0]);
        t.assert_equal((uint32_t) 0, cal->sampled_layer[1]);
        t.assert_equal((uint32_t) 4, cal->sampled_head[1]);
        t.assert_equal((uint32_t) 1, cal->sampled_layer[2]);
        t.assert_equal((uint32_t) 0, cal->sampled_head[2]);
        t.assert_equal((uint32_t) 1, cal->sampled_layer[3]);
        t.assert_equal((uint32_t) 4, cal->sampled_head[3]);
        triattention_calibration_free(cal);
        triattention_model_params_clear(&model);
    });

    return t.summary();
}
