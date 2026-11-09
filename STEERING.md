# Per-sequence control vectors (per-request steering)

Activation steering with a different control vector per sequence, set per request in `llama-server`.
Related upstream request: ggml-org/llama.cpp#10685.

Stock llama.cpp has one global control vector per context (`llama_set_adapter_cvec`, `--control-vector*`).
This branch adds a per-sequence table on top of it, so sequences decoded in the same batch can be steered
differently (or not at all) without splitting the batch.

## Core API (libllama)

```c
// llama_context_params
bool cvec_seq; // allocate per-sequence control vector tables (default false)

// set the control vector of one sequence, or clear it with data == NULL
int32_t llama_set_adapter_cvec_seq(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
                 const float * data,
                      size_t   len,
                     int32_t   n_embd,
                     int32_t   il_start,
                     int32_t   il_end);
```

- `data` has the same layout as `llama_set_adapter_cvec` / `common_control_vector_load`: `n_embd` floats per layer,
  starting at layer 1. Layers outside `[il_start, il_end]` are zero for this sequence.
- With `cvec_seq = true` the context allocates, per layer, an F32 tensor `[n_embd, n_seq_max + 1]` on the
  layer's buffer type. Row 0 is always zero, row `s + 1` belongs to sequence `s`. All rows start at zero.
- `llama_set_adapter_cvec_seq` only rewrites one row (one `ggml_backend_tensor_set` per layer). It does not change
  the graph and does not trigger a scheduler re-reserve (`test-cvec-seq` checks this). It synchronizes the context
  first, because the table may be read by a graph still in flight.
- Returns -1 if the context was created without `cvec_seq`, if `seq_id >= n_seq_max`, or if `n_embd` does not match.
- The per-sequence vector is added on top of the global control vector, so the two are additive.
- A token that belongs to more than one sequence (`n_seq_id > 1`, e.g. a shared prompt) uses row 0, which means it is not steered.

Memory: `(n_layer - 1) * n_embd * (n_seq_max + 1) * 4` bytes. For Bolt E4B (42 layers, n_embd 2560) that is
about 0.4 MiB per sequence, so 3.8 MiB at `-np 8`.

### Graph

`llm_graph_context::build_cvec(cur, il)` applies the global vector, then, when the table exists:

```
cur = ggml_add(cur, ggml_get_rows(table[il], idx))
```

`idx` is a new graph input (`llm_graph_input_cvec_seq`, I32), filled per ubatch from the token `seq_id`s in
`set_input()`, with `can_reuse()` so graph reuse keeps working. Once `cvec_seq` is enabled the get_rows+add is
always emitted, even when every row is zero, so the topology never depends on which sequences are steered.

`build_cvec` matches the index to the shape of the hidden state instead of changing each model builder:

- rows == n_tokens: one row per token (most models).
- rows == n_outputs: the builder cropped the hidden state to the output tokens (`inp_out_ids`) before this layer.
  This covers the Gemma 4 last-layer crop (`crop_before_nextn`) and the `prune_shared` crop of the KV-shared
  layers, Llama-style "skip computing output for unused tokens" crops on the last layer, and so on. A second index,
  `idx_out` (`[n_outputs]`), is built on the host in the same order as `llm_graph_input_out_ids`, so no
  `get_rows` on I32 is needed on the device.
- ne[2]*ne[3] == n_tokens (or n_outputs) with ne[1] > 1: several rows per token (DeepSeek-V4 hyper-connection
  streams). The vector is broadcast over ne[1], the same as the global vector.
- Anything else aborts with a clear message. The full arch sweep below hit no such case.

## Server

### Flags

| flag | |
|---|---|
| `--steering-vector NAME=FILE` | register a control vector GGUF (the `llama-cvector-generator` format) under `NAME`, can be repeated |
| `--steering` | enable per-request steering without startup vectors (vectors can then be added with `POST /steering-vectors`) |
| `--control-vector-layer-range START END` | also sets the layer range of the steering vectors (default: all layers) |

Steering is enabled (`cvec_seq = true` on the target context) when one of the first two flags is given. Files are
loaded at startup into CPU memory with their sha256. A file that does not load, or whose `n_embd` does not match
the model, stops the server.

### Endpoints

- `GET /steering-vectors`: `{"enabled": true, "vectors": [{"name", "path", "hash"}, ...]}`
- `POST /steering-vectors` with `{"name": "...", "path": "..."}`: registers (or replaces) a vector at runtime and
  returns `{"name", "path", "hash", "replaced"}`. Returns 400 for an unreadable or invalid file and 501 when steering
  is not enabled. The file is read on the HTTP thread, and the registry is mutex-guarded, so the main loop never
  waits on disk. Only CPU memory is touched.
  Security: this endpoint reads a path on the server host. Protect it like the rest of the API (`--api-key`, bind address).

### Request

Any completion endpoint (`/completion`, `/v1/completions`, `/v1/chat/completions`, ...) accepts:

```json
"steering": {"hunger": 0.6, "sleep": 0.2}
```

The slot mixes `sum(scale * vector)` once per task, on the CPU. Unknown names, non-numeric scales, or using
`steering` on a server without steering enabled return 400 before the task is queued. Scales of 0 are allowed,
echoed, and not mixed.

### Response echo

The response gets an extra top-level key. For streaming it is in the final chunk, the one with `"stop": true`
(native) or the last OAI chunk:

```json
"steering": {
  "hunger": {"scale": 0.6, "hash": "<sha256 of the file>"},
  "sleep":  {"scale": 0.2, "hash": "<sha256 of the file>"}
}
```

`generation_settings.steering` (native) also lists the requested scales. `/slots` shows `steering` for a slot that
is processing a steered task.

### When a slot is steered

- At launch the slot's mix is computed. Its row stays zero while its prompt is processed.
- When the slot first appears in the generation part of the batch (its prompt has been decoded and its first token
  sampled), its row is written: the mix, or zero if the request has no steering. Each generated token, including
  speculative draft tokens verified by the target, is then steered.
- On release the row is zeroed. If the task was steered, the KV of the generated tokens is removed
  (`llama_memory_seq_rm` from the first steered position, also on the draft context), and the slot's cached tokens
  are cut back to the prompt. Without this, a multi-turn request whose prompt is the previous prompt plus the
  previous reply would reuse KV computed under the previous request's steering. A cold server computes that reply
  unsteered, so the result would depend on cache state. Re-prefilling the reply is cheap. If the context was shifted
  during a steered generation, or partial removal is not supported (recurrent models), the slot's cache is cleared.
- Steering is not part of `server_slot::can_batch_with`, unlike LoRA. Slots with different steering decode in the
  same batch.

Consequences to be aware of:

- The first generated token is sampled from the logits of the last prompt token, which is unsteered. The server
  always re-evaluates at least one prompt token, even on a full cache hit. So single-token readouts
  (classification, "importance" scores, etc.) get no steering. Steering applies from the second generated token.
- The prompt is never steered, so prompt-cache reuse across requests with different steering is valid.

## Tests

### `tests/test-cvec-seq.cpp` (ctest `test-cvec-seq`)

Runs on the random-weight models generated by `test-llama-archs` (ctest fixture `generate-models`):
`gemma4-dense.gguf` (10 layers, 5 KV-shared, mixed SWA/full, per-layer input embeddings, so `prune_shared` and the
last-layer crop are live) and `llama-dense.gguf`. It runs with `GGML_SCHED_DEBUG_REALLOC=1`. It also takes
`-m model.gguf` (repeatable) or `--models DIR`. Each check runs with all-token outputs and with last-token-only
outputs (n_outputs < n_tokens), with `n_ubatch = 16`, so ubatches mix sequences.

| # | check | result (CPU) |
|---|---|---|
| 1 | zero table (with another sequence steered) vs stock context | bitwise equal |
| 2 | per-seq vector vs global `llama_set_adapter_cvec`, full and partial layer range | bitwise equal |
| 3 | two sequences with different vectors in one batch vs each alone | nmse 0 (bitwise) |
| 3b | same batch, flip only the other sequence's row | bitwise unchanged |
| 4 | `llama_state_seq_get_data` of a sequence prefilled unsteered next to a steered one, vs stock | bitwise equal |
| 5 | row updates between decodes | 0 re-reserves |

Also run outside ctest:

- All 145 models generated by `test-llama-archs` (`test-cvec-seq --models DIR`): 141 tested, 0 failures. The 4
  skipped are non-causal diffusion models (dream, llada, llada-moe, rnd1). On 4 archs the control vector has no
  effect, for the global vector as well (hy_v4, metis16, plamo2, qwen4exp: they do not call `build_cvec`).
- Tiny random models converted from HF with `convert_hf_to_gguf.py`:
  - Gemma 4: `Gemma4ForCausalLM`, 8 layers, hidden 256, `hidden_size_per_layer_input` 16, 3 KV-shared layers,
    `layer_types` S,S,S,S,F,S,S,F, double-wide MLP, real Gemma 4 tokenizer and chat template.
  - Llama: 4 layers, hidden 256, Llama-2 SPM tokenizer.

  All checks pass, bitwise.

### `tools/server/tests/unit/test_steering.py`

Uses `build/tests/test-models/gemma4-dense.gguf` by default (dummy tokenizer, so prompts are token ids). Override with
`LLAMA_STEERING_TEST_MODEL`. The steering vectors are written by the test with gguf-py. Greedy decoding, comparing
tokens and logprobs, never sampled text.

| test | |
|---|---|
| `test_no_steering_matches_stock` | no field, and all-zero scales, give output identical to a server without steering |
| `test_unknown_vector`, `test_not_enabled` | 400 |
| `test_concurrent_requests_get_their_own_steering` | 3 concurrent requests (2 steered differently, 1 unsteered) on 2 slots, two rounds, each matches its solo run (tokens exact, logprobs < 1e-3), echo correct |
| `test_steered_reply_is_not_reused_from_cache` | steered A, then unsteered B = A.prompt + A.reply on the same slot: only A.prompt is reused, and B matches a cold server. Verified to fail (27 tokens reused) with the cache trim disabled |
| `test_prompt_cache_reuse_with_different_steering` | prompt cache reused across different steering gives the cold result |
| `test_stream_and_oai_echo`, `test_chat_completion_echo` | echo in native/OAI responses and final stream chunks |
| `test_register_at_runtime` | GET/POST `/steering-vectors` |

Note on tolerances: in this llama.cpp, without any steering, the logits change by about 1e-4 when a request is
batched with others (batch composition), and by about 6e-5 when its prompt comes from the cache (the last prompt
token is re-evaluated in a different ubatch). Exact comparisons therefore disable the prompt cache. The concurrency
and cache tests compare tokens exactly, and logprobs with a 1e-3 tolerance, checked to be at least 10x below the
effect of the steering vector.

Results: 9/9 pass on `gemma4-dense.gguf`, `llama-dense.gguf`, the converted `gemma4-tiny.gguf` and `llama-tiny.gguf`.

`ctest` (CPU build): all tests pass except the ones that need network access, which this environment does not
have (huggingface.co is blocked): `test-download-model`, `test-eval-callback-download-model`, the
download section of `test-arg-parser`, and the dependent `test-llama-context-model`, `test-thread-safety` and
`test-eval-callback` (not run).

The server test harness downloads its HF presets at session start (`ServerPreset.load_all`), which fails offline.
To run `test_steering.py` without network, stub that out:

```bash
cd tools/server/tests
python3 -c "import sys, utils, pytest; utils.ServerPreset.load_all = staticmethod(lambda: None); sys.exit(pytest.main(['-q', 'unit/test_steering.py']))"
```

### Speculative decoding (CPU sanity check)

With `-md` the same tiny Llama model as its own draft (`--spec-type draft-simple`), greedy steered and unsteered
outputs are identical to the non-speculative runs. Draft acceptance is 41/41 unsteered and 0/340 with a strong
(scale 2, random) steering vector, because the draft model is not steered. ngram speculation produced no drafts on
this random text.

## Not verified yet (GPU stage, AMD R9700 / Vulkan)

1. **Vulkan correctness**: run `test-cvec-seq` with the Vulkan backend (`-m` the real Bolt Q4 GGUF and the generated
   models) and `test_steering.py` against a Vulkan build. Bitwise equality with the global path is expected but
   not guaranteed if Vulkan fuses the global broadcast add differently. Confirm that `GET_ROWS` with an F32 source
   and an I32 index is supported and not falling back to CPU (check `GGML_SCHED` logs / graph splits).
2. **Performance**:
   - Decode throughput with `cvec_seq` on vs off (one extra get_rows + add per layer).
   - The cost of a row write: 41 small `ggml_backend_tensor_set` uploads plus a `synchronize()` per request start
     and release. On Vulkan each may be a staging copy and stall. If it matters, upload the whole row in one
     staging copy (e.g. a contiguous `[n_embd, n_layer]` per-row layout) or batch the writes of all slots per
     update_slots iteration.
   - Batching: check that slots with different steering still share batches under load (`/slots`, timing).
3. **Speculative decoding with steering**: draft tokens are steered when the target verifies them, but the draft
   model (or MTP/DFlash head) is not steered, so expect a lower acceptance rate as steering gets stronger (0% for a
   strong random vector above). Measure with the real model and real drives vectors.
4. **Real model**: the Bolt E4B Q4_K_M GGUF (`ggml-org/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF`) could not be downloaded
   here (no HF access). Run `test-cvec-seq -m <bolt.gguf>` and the server tests with `LLAMA_STEERING_TEST_MODEL`.
5. Not covered: `-sm row`/tensor split across several GPUs (the table follows `model.select_buft(il)` like the
   global vector), multimodal prompts (positions vs token counts in the cache trim are handled via `pos_next()` but
   untested), and recurrent/hybrid models in the server (they fall back to clearing the slot cache after a steered
   request).
