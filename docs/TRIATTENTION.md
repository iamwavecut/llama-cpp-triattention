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
| Can TriAttention run without an external `.triattention` file? | Yes. The runtime checks embedded `GGUF` calibration first, then an explicit file, then a sidecar, then fallback |
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

### 1. Build a calibrated `GGUF` copy

The calibration tool consumes a local `GGUF` model and a plain-text corpus:

```bash
./build/bin/llama-triattention-calibrate \
    -m models/model.gguf \
    -f corpus.txt \
    -o models/model.triattention.gguf \
    -c 8192 \
    -b 2048
```

Optional external-artifact mode:

```bash
./build/bin/llama-triattention-calibrate \
    -m models/model.gguf \
    -f corpus.txt \
    --external-out models/model.triattention \
    --no-embed \
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

### 2. Inspect the output

```bash
./build/bin/llama-triattention-calibrate --inspect models/model.triattention.gguf
./build/bin/llama-triattention-calibrate --inspect models/model.triattention
```

This prints the file version, model metadata, head counts, frequency count,
and whether explicit `omega` / `freq_scale_sq` arrays are embedded.

### 3. Validate against a model

```bash
./build/bin/llama-triattention-calibrate \
    --validate models/model.triattention.gguf
```

Validation checks the model geometry and warns on RoPE parameter mismatches.
For an external artifact:

```bash
./build/bin/llama-triattention-calibrate \
    --validate models/model.triattention \
    -m models/model.gguf
```

## Inference workflow

### Calibrated mode

```bash
./build/bin/llama-server \
    -m models/model.triattention.gguf \
    --triattention-budget 2048 \
    --triattention-window 128 \
    --triattention-trigger interval \
    -c 131072
```

Explicit external-artifact mode:

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
| `--triattention-stats PATH` | none | Explicit external calibration artifact. Checked only after embedded `GGUF` calibration |
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
| Loaded `GGUF` contains embedded calibration | Embedded calibrated TriAttention is used |
| No embedded calibration, `--triattention-stats` points to a valid file | External calibrated TriAttention is used |
| No embedded calibration and no explicit file, but `<model>.triattention` exists and is valid | Sidecar calibrated TriAttention is used |
| Sidecar file exists but is invalid or incompatible | It is ignored with a warning |
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
| `2` | Read/write compatibility | Uniform-layout explicit-rope format |
| `3` | Current write format | Per-layer geometry and RoPE metadata; default serialized artifact format |

### Header

```text
magic          u32    0x54524941
version        u32    1, 2, or 3
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

Version 2 appends:

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

Version 3 additionally stores per-layer head geometry, RoPE layout, KV-source
mapping, and optional explicit `omega` / `freq_scale_sq` arrays per layer so
heterogeneous models such as Gemma 4 / iSWA can be calibrated faithfully.

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
