# TriAttention API Reference

## Public C API

### `llama_triattention_init`

```c
LLAMA_API int32_t llama_triattention_init(
        struct llama_context * ctx,
                  const char * stats_path,
                     int32_t   budget,
                     int32_t   divide_length,
                     int32_t   offset_max,
                     int32_t   mode,
                     int32_t   trigger,
                     int32_t   agg,
                     int32_t   seed,
                        bool   normalize_scores,
                        bool   protect_prefill,
                        bool   disable_mlr,
                        bool   disable_trig,
                        bool   enable_logging,
                     int32_t   fallback_mode,
                        float   fallback_recency_weight);
```

Initializes TriAttention on a `llama_context`.

| Parameter | Description |
|-----------|-------------|
| `ctx` | Context whose KV cache will be pruned |
| `stats_path` | Optional explicit path to an external `.triattention` file |
| `budget` | Maximum KV entries retained after pruning |
| `divide_length` | Pruning interval; the same recent window is protected from eviction |
| `offset_max` | Maximum geometric offset for trig scoring |
| `mode` | `0=global`, `1=per-kv-head`, `2=per-layer-head` |
| `trigger` | `0=interval`, `1=slack` |
| `agg` | `0=mean`, `1=max` |
| `seed` | Tie-breaking noise seed; `-1` disables it |
| `normalize_scores` | Z-score normalize per-head scores before selection |
| `protect_prefill` | Keep prompt/prefix tokens protected |
| `disable_mlr` | Disable MLR weighting in the norm term |
| `disable_trig` | Disable the trigonometric term |
| `enable_logging` | Log prune events to stderr |
| `fallback_mode` | `0=off`, `1=auto`, `2=hybrid-norm-recency` |
| `fallback_recency_weight` | Blend factor for fallback recency scoring |

Returns `0` on success and `-1` on failure.

## Internal runtime API

### Core lifecycle

```c
triattention_state * triattention_init(
    const char * stats_path,
    const triattention_config * cfg,
    const triattention_model_params * model,
    const uint32_t * sampled_layers,
    uint32_t n_sampled_layers);

triattention_state * triattention_init_from_calibration(
    const triattention_calibration * calibration,
    const char * source_name,
    const triattention_config * cfg,
    const triattention_model_params * model,
    const uint32_t * sampled_layers,
    uint32_t n_sampled_layers);

void triattention_free(triattention_state * state);
```

`triattention_init()` resolves calibration in this order:

1. Embedded calibration inside the loaded `GGUF`
2. Explicit `stats_path`
3. Sidecar `<model>.triattention`
4. Runtime fallback, if enabled

### Scoring and pruning helpers

```c
void triattention_invert_rope(
    float * k,
    const float * omega,
    const int32_t * positions,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t rope_style);

void triattention_score_keys(
    float * out_scores,
    const float * pre_rope_k,
    const triattention_head_stats * stats,
    const float * omega,
    const float * freq_scale_sq,
    const float * offsets,
    const int32_t * key_positions,
    int64_t round_start,
    uint32_t n_keys,
    uint32_t head_dim,
    uint32_t freq_count,
    uint32_t n_offsets,
    enum triattention_agg agg,
    bool disable_trig);

void triattention_build_recency_scores(
    float * out_scores,
    const int32_t * key_positions,
    uint32_t n_keys);

void triattention_blend_fallback_scores(
    float * scores,
    const float * recency_scores,
    uint32_t n_keys,
    float lambda);

void triattention_prune_impl(
    triattention_state * state,
    const ggml_tensor ** k_tensors,
    const uint32_t * layer_map,
    uint32_t n_layers,
    uint32_t n_kv_heads,
    uint32_t padded_head_dim,
    uint32_t kv_size,
    uint32_t * evicted_cells,
    uint32_t * n_evicted);
```

## Internal file API

`src/llama-triattention-file.cpp` owns the binary `.triattention` format:

```c
bool triattention_build_rope_arrays(
    float * omega,
    float * freq_scale_sq,
    uint32_t freq_count,
    const triattention_rope_params * params);

triattention_calibration * triattention_calibration_load(const char * path, bool verbose);
triattention_calibration * triattention_calibration_load_from_buffer(
    const void * data,
    size_t size,
    bool verbose,
    const char * source_name);
bool triattention_calibration_save(const char * path, const triattention_calibration * cal);
bool triattention_calibration_save_to_buffer(
    const triattention_calibration * cal,
    std::vector<uint8_t> & out);
void triattention_calibration_free(triattention_calibration * cal);

bool triattention_calibration_validate(
    const triattention_calibration * cal,
    const triattention_model_params * model,
    bool warn_rope_theta);

triattention_calibration * triattention_calibration_create_fallback(
    const triattention_model_params * model);
```

## Internal calibration API

The standalone tool uses `ggml_backend_sched_eval_callback()` and the helpers
below to collect pre-RoPE query statistics directly from a `GGUF` model:

```c
bool triattention_is_query_rope_name(const char * name);
bool triattention_parse_layer_index(const char * name, uint32_t * layer_idx);
bool triattention_extract_rope_params(
    const ggml_tensor * rope_tensor,
    triattention_rope_params * params,
    uint32_t * rope_style,
    std::string * error);

class triattention_calibration_builder {
public:
    triattention_calibration_builder(
        std::string model_name,
        uint32_t num_layers,
        uint32_t num_attn_heads,
        uint32_t num_kv_heads);

    void note_rope_source(const char * name);

    bool accumulate_query_tensor(
        const ggml_tensor * src0,
        const void * src0_data,
        uint32_t layer_idx,
        uint32_t rope_style,
        const triattention_rope_params & rope_params,
        std::string * error);

    triattention_calibration * finalize(std::string * error) const;
};
```

The builder computes streaming statistics for each sampled `(layer, attn_head,
freq)` tuple:

| Field | Meaning |
|-------|---------|
| `q_mean_real` | Real part of `E[q_f]` |
| `q_mean_imag` | Imaginary part of `E[q_f]` |
| `q_abs_mean` | `E[|q_f|]` |
| `r_f` | `|E[q_f]| / E[|q_f|]` |

## Core structs

### `triattention_model_params`

```c
struct triattention_model_params {
    uint32_t kv_size;
    uint32_t head_dim;
    uint32_t num_layers;
    uint32_t num_attn_heads;
    uint32_t num_kv_heads;
    uint32_t rope_style;
    uint32_t n_ctx_orig;

    double rope_theta;
    float  rope_freq_scale;
    float  rope_ext_factor;
    float  rope_attn_factor;
    float  rope_beta_fast;
    float  rope_beta_slow;
};
```

### `triattention_config`

```c
struct triattention_config {
    uint32_t budget;
    uint32_t divide_length;
    uint32_t offset_max;

    enum triattention_mode    mode;
    enum triattention_trigger trigger;
    enum triattention_agg     agg;

    bool normalize_scores;
    bool protect_prefill;
    bool disable_mlr;
    bool disable_trig;
    bool enable_logging;

    int32_t seed;

    enum triattention_fallback fallback_mode;
    float fallback_recency_weight;
};
```

### `triattention_calibration`

```c
struct triattention_calibration {
    uint32_t version;
    uint32_t head_dim;
    uint32_t num_layers;
    uint32_t num_attn_heads;
    uint32_t num_kv_heads;
    uint32_t num_kv_groups;
    double   rope_theta;
    uint32_t rope_style;
    uint32_t freq_count;
    uint32_t n_sampled;

    float * omega;
    float * freq_scale_sq;

    uint32_t * sampled_layer;
    uint32_t * sampled_head;
    triattention_head_stats * head_stats;

    char model_name[256];
};
```

## Enums

### `triattention_mode`

| Value | Name |
|-------|------|
| `0` | `TRIATTENTION_MODE_GLOBAL` |
| `1` | `TRIATTENTION_MODE_PER_KV_HEAD` |
| `2` | `TRIATTENTION_MODE_PER_LAYER_HEAD` |

### `triattention_trigger`

| Value | Name |
|-------|------|
| `0` | `TRIATTENTION_TRIGGER_INTERVAL` |
| `1` | `TRIATTENTION_TRIGGER_SLACK` |

### `triattention_agg`

| Value | Name |
|-------|------|
| `0` | `TRIATTENTION_AGG_MEAN` |
| `1` | `TRIATTENTION_AGG_MAX` |

### `triattention_fallback`

| Value | Name |
|-------|------|
| `0` | `TRIATTENTION_FALLBACK_OFF` |
| `1` | `TRIATTENTION_FALLBACK_AUTO` |
| `2` | `TRIATTENTION_FALLBACK_HYBRID_NORM_RECENCY` |

## User-facing tool

The supported calibration entrypoint is the standalone binary:

```text
llama-triattention-calibrate
```

Supported modes:

| Mode | Example |
|------|---------|
| Build embedded `GGUF` | `llama-triattention-calibrate -m model.gguf -f corpus.txt -o model.triattention.gguf` |
| Build external artifact | `llama-triattention-calibrate -m model.gguf -f corpus.txt --external-out model.triattention --no-embed` |
| Inspect | `llama-triattention-calibrate --inspect model.triattention.gguf` |
| Validate | `llama-triattention-calibrate --validate model.triattention.gguf` |
