#include "llama-triattention-file.h"
#include "llama-triattention.h"
#include "testing.h"

#include <cmath>

static bool approx_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
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
        triattention_model_params model = {};
        model.head_dim = 8;
        model.num_layers = 2;
        model.num_attn_heads = 8;
        model.num_kv_heads = 2;
        model.rope_style = 0;
        model.rope_theta = 10000.0;

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
    });

    return t.summary();
}
