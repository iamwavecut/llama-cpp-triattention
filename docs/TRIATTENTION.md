# TriAttention: Calibration, Runtime, and File Format

## Overview

TriAttention is a KV-cache pruning system based on the paper
["TriAttention: Trigonometric KV Cache Eviction"](https://arxiv.org/abs/2604.04921).
It uses RoPE-inverted key vectors and, in the canonical path, offline query
statistics collected from the same model family.

This repository now supports two runtime modes:

| Mode | Status | Equivalent to the paper |
|------|--------|-------------------------|
| `calibrated` | Recommended | Yes, this is the intended path |
| `experimental fallback` | Optional | No, this is a heuristic runtime-only mode |

## Do you need calibration?

| Question | Answer |
|----------|--------|
| Can TriAttention run without a `.triattention` file? | Yes, if `--triattention-fallback auto` or `--triattention-fallback hybrid-norm-recency` is enabled |
| Is that equivalent to the paper? | No |
| What should be used for quality-sensitive runs? | A calibration file built from a representative corpus and the same target model |

The paper-aligned method depends on offline statistics of pre-RoPE query
activations. That part cannot be removed without changing the algorithm.

## Build

Build the server, CLI, and standalone calibration tool:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cli llama-server llama-triattention-calibrate -j
```

## Calibration workflow

### 1. Build a `.triattention` file

The calibration tool consumes a local `GGUF` model and a plain-text corpus:

```bash
./build/bin/llama-triattention-calibrate \
    -m models/model.gguf \
    -f corpus.txt \
    -o models/model.triattention \
    -c 8192 \
    -b 2048
```

Notes:

| Item | Detail |
|------|--------|
| Corpus format | Plain text file |
| Tokenization | Uses the model's own tokenizer |
| Chunking | The corpus is processed in chunks of `-c` / `n_ctx` |
| KV handling | KV cache is cleared between chunks |
| Capture point | Query tensors are captured via `ggml_backend_sched_eval_callback()` on `GGML_OP_ROPE` inputs |

### 2. Inspect the file

```bash
./build/bin/llama-triattention-calibrate --inspect models/model.triattention
```

This prints the file version, model metadata, head counts, frequency count,
and whether explicit `omega` / `freq_scale_sq` arrays are embedded.

### 3. Validate against a model

```bash
./build/bin/llama-triattention-calibrate \
    --validate models/model.triattention \
    -m models/model.gguf
```

Validation checks the model geometry and warns on RoPE parameter mismatches.

## Inference workflow

### Calibrated mode

```bash
./build/bin/llama-server \
    -m models/model.gguf \
    --triattention-stats models/model.triattention \
    --triattention-budget 2048 \
    --triattention-window 128 \
    --triattention-trigger interval \
    -c 131072
```

### Experimental fallback mode

```bash
./build/bin/llama-server \
    -m models/model.gguf \
    --triattention-fallback auto \
    --triattention-fallback-recency-weight 0.25 \
    --triattention-budget 2048 \
    --triattention-window 128 \
    -c 131072
```

Fallback mode is useful when you want pruning to work immediately without an
offline calibration pass, but it should be treated as a separate heuristic.

## Runtime flags

| Flag | Default | Description |
|------|---------|-------------|
| `--triattention-stats PATH` | none | Path to the calibration file; preferred runtime path |
| `--triattention-budget N` | `512` | Maximum KV entries retained after pruning |
| `--triattention-window N` | `64` | Pruning interval in decode tokens; the same recent window is protected from eviction |
| `--triattention-offset-max N` | `65536` | Maximum geometric offset used by trig scoring |
| `--triattention-mode MODE` | `global` | `global`, `per-kv-head`, or `per-layer-head` |
| `--triattention-trigger MODE` | `interval` | `interval` or `slack` |
| `--triattention-agg MODE` | `mean` | `mean` or `max` aggregation over offsets |
| `--triattention-seed N` | `0` | Tie-breaking noise seed; `-1` disables it |
| `--triattention-normalize` | off | Z-score normalize per-head scores before selection |
| `--triattention-no-protect-prefill` | off | Allow prompt/prefix tokens to be evicted |
| `--triattention-disable-mlr` | off | Ablation: disable MLR weighting in the norm term |
| `--triattention-disable-trig` | off | Ablation: drop the trigonometric term |
| `--triattention-log` | off | Log prune events to stderr |
| `--triattention-fallback MODE` | `auto` | `auto`, `off`, or `hybrid-norm-recency` |
| `--triattention-fallback-recency-weight F` | `0.25` | Blend factor for fallback recency scoring |

## Fallback behavior

| Situation | Result |
|-----------|--------|
| `--triattention-stats` points to a valid file | Calibrated TriAttention is used |
| Stats file is missing or not provided and fallback is `auto` | Runtime constructs a fallback state and continues |
| Stats file is provided but invalid or incompatible | Hard error |
| No stats file and fallback is `off` | Hard error |

The fallback scorer reuses the same RoPE-inverted key path and ranks tokens by
a normalized blend of:

1. Key norm score weighted by `freq_scale_sq`
2. Recency score over the current prune candidate set

## Calibration file format

The binary format uses magic `0x54524941` (`TRIA`).

| Version | Status | Notes |
|---------|--------|-------|
| `1` | Read-only compatibility | Legacy files without explicit `omega` / `freq_scale_sq` arrays |
| `2` | Current write format | Default output of `llama-triattention-calibrate` |

### Header

```text
magic          u32    0x54524941
version        u32    1 or 2
head_dim       u32
num_layers     u32
num_attn_heads u32
num_kv_heads   u32
rope_theta     f64
rope_style     u32
n_sampled      u32
freq_count     u32
name_len       u32
name           char[name_len]
```

Version 2 then appends:

```text
omega          f32[freq_count]
freq_scale_sq  f32[freq_count]
```

Per sampled head:

```text
layer_idx      u32
head_idx       u32
q_mean_real    f32[freq_count]
q_mean_imag    f32[freq_count]
q_abs_mean     f32[freq_count]
r_f            f32[freq_count]
```

For version 1 files, runtime reconstructs `omega` from the model's RoPE
parameters and uses `freq_scale_sq = 1` when explicit arrays are absent.

## Source layout

| File | Purpose |
|------|---------|
| `src/llama-triattention.cpp` | Runtime scoring, pruning, and fallback logic |
| `src/llama-triattention.h` | Core structs, enums, and internal APIs |
| `src/llama-triattention-file.cpp` | `.triattention` load/save/validate helpers |
| `src/llama-triattention-calibration.cpp` | Calibration capture helpers and streaming statistics |
| `tools/triattention-calibrate/triattention-calibrate.cpp` | Standalone calibration tool |
| `src/llama-kv-cache.cpp` | KV hooks that trigger pruning and track cell positions |
| `common/arg.cpp` | CLI/server flag registration |

## Current limitations

| Limitation | Detail |
|------------|--------|
| Corpus mode | Plain text only in the first implementation |
| Model family | Decoder-only RoPE models whose query path appears as `GGML_OP_ROPE` over 3D query tensors |
| Fallback semantics | Heuristic only; not a paper claim |

