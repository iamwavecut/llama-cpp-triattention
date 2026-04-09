#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml-backend.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-triattention-calibration.h"
#include "llama-triattention-file.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

enum class tool_mode {
    build,
    inspect,
    validate,
};

struct capture_context {
    explicit capture_context(triattention_calibration_builder & builder) : builder(builder) {}

    triattention_calibration_builder & builder;
    std::vector<char> src0_buf;
    std::vector<float> freq_factor_buf;
    std::string error;
    bool failed = false;
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f corpus.txt -o model.triattention [-c 8192] [-b 2048]\n", argv[0]);
    LOG("    %s --inspect model.triattention\n", argv[0]);
    LOG("    %s --validate model.triattention -m model.gguf\n", argv[0]);
    LOG("\n");
}

static bool triattention_capture_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * ctx = static_cast<capture_context *>(user_data);
    if (!t || t->op != GGML_OP_ROPE || t->src[0] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = t->src[0];
    ctx->builder.note_rope_source(src0->name);

    if (!triattention_is_query_rope_name(src0->name)) {
        return false;
    }

    if (ask) {
        return src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16;
    }

    if (ctx->failed) {
        return true;
    }

    uint32_t layer_idx = 0;
    if (!triattention_parse_layer_index(src0->name, &layer_idx)) {
        ctx->error = std::string("failed to parse layer index from tensor name: ") + src0->name;
        ctx->failed = true;
        return true;
    }

    triattention_rope_params rope_params = {};
    uint32_t rope_style = 0;
    if (!triattention_extract_rope_params(t, &rope_params, &rope_style, &ctx->error)) {
        ctx->failed = true;
        return true;
    }

    if (t->src[2] != nullptr) {
        const ggml_tensor * src2 = t->src[2];
        const bool src2_is_host = src2->buffer == nullptr || ggml_backend_buffer_is_host(src2->buffer);
        if (src2_is_host) {
            rope_params.freq_factors = static_cast<const float *>(src2->data);
            rope_params.freq_factor_count = (uint32_t) src2->ne[0];
        } else {
            ctx->freq_factor_buf.resize((size_t) src2->ne[0]);
            ggml_backend_tensor_get(src2, ctx->freq_factor_buf.data(), 0, sizeof(float) * (size_t) src2->ne[0]);
            rope_params.freq_factors = ctx->freq_factor_buf.data();
            rope_params.freq_factor_count = (uint32_t) src2->ne[0];
        }
    }

    const bool src0_is_host = src0->buffer == nullptr || ggml_backend_buffer_is_host(src0->buffer);
    const void * src0_data = src0->data;
    if (!src0_is_host) {
        const size_t nbytes = ggml_nbytes(src0);
        ctx->src0_buf.resize(nbytes);
        ggml_backend_tensor_get(src0, ctx->src0_buf.data(), 0, nbytes);
        src0_data = ctx->src0_buf.data();
    }

    if (!ctx->builder.accumulate_query_tensor(src0, src0_data, layer_idx, rope_style, rope_params, &ctx->error)) {
        ctx->failed = true;
    }

    return true;
}

static bool build_model_params_from_context(llama_context * ctx, triattention_model_params * out) {
    if (!ctx || !out) {
        return false;
    }

    const llama_model & model = ctx->get_model();
    const llama_cparams & cparams = ctx->get_cparams();
    return triattention_model_params_init(&model, &cparams, 0, out);
}

static std::string infer_model_name(const llama_model * model, const common_params & params) {
    char buf[256];
    if (model && llama_model_meta_val_str(model, "general.name", buf, sizeof(buf)) > 0 && buf[0] != '\0') {
        return buf;
    }

    if (!params.model.path.empty()) {
        const size_t pos = params.model.path.find_last_of("/\\");
        return pos == std::string::npos ? params.model.path : params.model.path.substr(pos + 1);
    }
    if (!params.model.hf_repo.empty()) {
        return params.model.hf_repo;
    }
    return "unknown-model";
}

static int run_inspect(const std::string & path) {
    triattention_calibration * cal = triattention_calibration_load(path.c_str(), false);
    if (!cal) {
        return 1;
    }

    double mean_rf = 0.0;
    uint64_t rf_count = 0;
    for (uint32_t i = 0; i < cal->n_sampled; ++i) {
        if (!cal->head_stats[i].r_f) {
            continue;
        }
        for (uint32_t f = 0; f < cal->freq_count; ++f) {
            mean_rf += cal->head_stats[i].r_f[f];
            ++rf_count;
        }
    }

    LOG("file:               %s\n", path.c_str());
    LOG("version:            %u\n", cal->version);
    LOG("model:              %s\n", cal->model_name);
    LOG("layers:             %u\n", cal->num_layers);
    LOG("attention heads:    %u\n", cal->num_attn_heads);
    LOG("kv heads:           %u\n", cal->num_kv_heads);
    LOG("head dim:           %u\n", cal->head_dim);
    LOG("freq count:         %u\n", cal->freq_count);
    LOG("sampled heads:      %u\n", cal->n_sampled);
    LOG("rope theta:         %.1f\n", cal->rope_theta);
    LOG("rope style:         %s\n", cal->rope_style == 0 ? "half" : "interleaved");
    LOG("explicit omega:     %s\n", cal->omega ? "yes" : "no");
    LOG("explicit fscale^2:  %s\n", cal->freq_scale_sq ? "yes" : "no");
    if (rf_count > 0) {
        LOG("mean r_f:           %.6f\n", mean_rf / (double) rf_count);
    }

    triattention_calibration_free(cal);
    return 0;
}

static bool run_calibration(llama_context * ctx, const common_params & params, capture_context & capture) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    const int32_t chunk_size = params.n_ctx > 0 ? params.n_ctx : (int32_t) llama_n_ctx(ctx);
    const int32_t batch_size = std::max<int32_t>(1, std::min(params.n_batch, chunk_size));

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);
    if (tokens.empty()) {
        LOG_ERR("%s: calibration corpus is empty after tokenization\n", __func__);
        return false;
    }

    LOG_INF("%s: tokenized %zu tokens, chunk_size=%d, batch_size=%d\n",
            __func__, tokens.size(), chunk_size, batch_size);

    if (params.i_chunk > 0) {
        const size_t skip_tokens = (size_t) params.i_chunk * (size_t) chunk_size;
        if (skip_tokens >= tokens.size()) {
            LOG_ERR("%s: there are not enough tokens left after removing %d chunks (%zu tokens)\n",
                    __func__, params.i_chunk, skip_tokens);
            return false;
        }

        LOG_INF("%s: removing initial %d chunks (%zu tokens)\n",
                __func__, params.i_chunk, skip_tokens);
        tokens.erase(tokens.begin(), tokens.begin() + (ptrdiff_t) skip_tokens);
    }

    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    const size_t n_chunk_max = (tokens.size() + (size_t) chunk_size - 1) / (size_t) chunk_size;
    const size_t n_chunk = params.n_chunks < 0
        ? n_chunk_max
        : std::min((size_t) params.n_chunks, n_chunk_max);

    LOG_INF("%s: computing over %zu chunks\n", __func__, n_chunk);

    for (size_t chunk = 0; chunk < n_chunk; ++chunk) {
        const size_t start = chunk * (size_t) chunk_size;
        const size_t end = std::min(tokens.size(), start + (size_t) chunk_size);

        llama_memory_clear(llama_get_memory(ctx), true);

        for (size_t batch_start = start; batch_start < end; batch_start += (size_t) batch_size) {
            const int cur_batch = (int) std::min((size_t) batch_size, end - batch_start);
            common_batch_clear(batch);

            for (int i = 0; i < cur_batch; ++i) {
                llama_token tok = tokens[batch_start + (size_t) i];
                if (add_bos && batch_start == start && i == 0) {
                    tok = llama_vocab_bos(vocab);
                }
                common_batch_add(batch, tok, (llama_pos) ((batch_start - start) + (size_t) i), { 0 }, false);
            }

            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: llama_decode failed on chunk %zu\n", __func__, chunk);
                llama_batch_free(batch);
                return false;
            }
            if (capture.failed) {
                LOG_ERR("%s: calibration capture failed: %s\n", __func__, capture.error.c_str());
                llama_batch_free(batch);
                return false;
            }
        }
    }

    llama_batch_free(batch);
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    tool_mode mode = tool_mode::build;
    std::string target_file;
    std::vector<char *> filtered_argv;
    filtered_argv.reserve((size_t) argc);
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--inspect" || arg == "--validate") {
            if (!target_file.empty()) {
                LOG_ERR("only one of --inspect/--validate may be used\n");
                return 1;
            }
            if (i + 1 >= argc) {
                LOG_ERR("missing file path after %s\n", arg.c_str());
                return 1;
            }
            mode = arg == "--inspect" ? tool_mode::inspect : tool_mode::validate;
            target_file = argv[++i];
            continue;
        }
        filtered_argv.push_back(argv[i]);
    }

    if (mode == tool_mode::inspect) {
        return run_inspect(target_file);
    }

    if (!common_params_parse((int) filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.warmup = false;
    params.compute_ppl = false;
    params.n_parallel = 1;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    if (mode == tool_mode::validate) {
        if (params.model.path.empty()) {
            LOG_ERR("validation requires -m <model.gguf>\n");
            llama_backend_free();
            return 1;
        }

        auto init = common_init_from_params(params);
        llama_context * ctx = init->context();
        if (!ctx) {
            LOG_ERR("failed to initialize model for validation\n");
            llama_backend_free();
            return 1;
        }

        triattention_model_params model_params = {};
        if (!build_model_params_from_context(ctx, &model_params)) {
            LOG_ERR("failed to derive model parameters for validation\n");
            llama_backend_free();
            return 1;
        }

        triattention_calibration * cal = triattention_calibration_load(target_file.c_str(), false);
        if (!cal) {
            llama_backend_free();
            return 1;
        }

        const bool ok = triattention_calibration_validate(cal, &model_params, true);
        LOG("%s\n", ok ? "validation: OK" : "validation: FAILED");
        triattention_calibration_free(cal);
        triattention_model_params_clear(&model_params);
        llama_backend_free();
        return ok ? 0 : 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("calibration build requires -m <model.gguf>\n");
        llama_backend_free();
        return 1;
    }
    if (params.prompt.empty()) {
        LOG_ERR("calibration build requires -f <corpus.txt>\n");
        llama_backend_free();
        return 1;
    }
    if (params.out_file.empty()) {
        LOG_ERR("calibration build requires -o <output.triattention>\n");
        llama_backend_free();
        return 1;
    }

    triattention_model_params model_params = {};
    triattention_calibration_builder builder("", nullptr);
    capture_context capture(builder);
    params.cb_eval = triattention_capture_cb;
    params.cb_eval_user_data = &capture;

    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    llama_model * model = init->model();
    if (!ctx || !model) {
        LOG_ERR("failed to initialize model/context for calibration\n");
        llama_backend_free();
        return 1;
    }

    if (!triattention_model_params_init(model, &ctx->get_cparams(), 0, &model_params)) {
        LOG_ERR("failed to derive model parameters for calibration\n");
        llama_backend_free();
        return 1;
    }

    builder = triattention_calibration_builder(infer_model_name(model, params), &model_params);

    capture_context real_capture(builder);
    params.cb_eval_user_data = &real_capture;

    // Rebind callback user data on the live context scheduler.
    ggml_backend_sched_set_eval_callback(ctx->get_sched(), triattention_capture_cb, &real_capture);

    const bool ok = run_calibration(ctx, params, real_capture);
    if (!ok) {
        if (!builder.has_captured_queries()) {
            LOG_ERR("seen rope sources:\n");
            for (const auto & name : builder.rope_sources_seen()) {
                LOG_ERR("  %s\n", name.c_str());
            }
        }
        triattention_model_params_clear(&model_params);
        llama_backend_free();
        return 1;
    }

    std::string error;
    triattention_calibration * cal = builder.finalize(&error);
    if (!cal) {
        LOG_ERR("failed to finalize calibration: %s\n", error.c_str());
        LOG_ERR("seen rope sources:\n");
        for (const auto & name : builder.rope_sources_seen()) {
            LOG_ERR("  %s\n", name.c_str());
        }
        triattention_model_params_clear(&model_params);
        llama_backend_free();
        return 1;
    }

    if (!triattention_calibration_save(params.out_file.c_str(), cal)) {
        triattention_calibration_free(cal);
        triattention_model_params_clear(&model_params);
        llama_backend_free();
        return 1;
    }

    LOG("wrote calibration: %s\n", params.out_file.c_str());
    triattention_calibration_free(cal);
    triattention_model_params_clear(&model_params);
    llama_backend_free();
    return 0;
}
