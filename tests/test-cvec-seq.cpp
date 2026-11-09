// per-sequence control vectors (llama_set_adapter_cvec_seq)
//
// usage:
//   test-cvec-seq -m model.gguf [-m model2.gguf ...]
//   test-cvec-seq --models DIR     (all *.gguf in DIR, models that fail the stock run are skipped)
//
// checks, per model:
//   1. an allocated but zero table gives the same logits as a stock context
//   2. a vector set on one sequence matches the global control vector (llama_set_adapter_cvec)
//   3. two sequences with different vectors in one batch match each sequence decoded alone
//   4. an unsteered sequence decoded in the same batch as a steered one has the same state as without steering
//   5. updating the table does not trigger a graph re-reserve

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

static std::atomic<int> g_n_reserve{0};
static bool g_verbose = false;

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (strstr(text, "reserving ...") != nullptr) {
        g_n_reserve++;
    }
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

struct seq_prompt {
    llama_seq_id             seq_id;
    std::vector<llama_token> tokens;
};

// decode all prompts in one llama_decode call (split into ubatches by the context)
// returns the logits of the output tokens, in batch order. output = all tokens or only the last token of each prompt
static bool decode(llama_context * ctx, const std::vector<seq_prompt> & prompts, bool all_logits, std::vector<float> & out) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    int n_tokens = 0;
    for (const auto & p : prompts) {
        n_tokens += p.tokens.size();
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (const auto & p : prompts) {
        const llama_pos pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx), p.seq_id) + 1;
        for (size_t i = 0; i < p.tokens.size(); ++i) {
            const int j = batch.n_tokens++;
            batch.token   [j]    = p.tokens[i];
            batch.pos     [j]    = pos0 + i;
            batch.n_seq_id[j]    = 1;
            batch.seq_id  [j][0] = p.seq_id;
            batch.logits  [j]    = all_logits || i + 1 == p.tokens.size();
        }
    }

    const int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        llama_batch_free(batch);
        return false;
    }

    out.clear();
    for (int j = 0; j < batch.n_tokens; ++j) {
        if (batch.logits[j]) {
            const float * l = llama_get_logits_ith(ctx, j);
            out.insert(out.end(), l, l + n_vocab);
        }
    }

    llama_batch_free(batch);
    return true;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    double err = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        err += (a[i] - b[i])*(a[i] - b[i]);
        ref += a[i]*a[i];
    }
    return ref > 0.0 ? err/ref : err;
}

static std::vector<uint8_t> seq_state(llama_context * ctx, llama_seq_id seq_id) {
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, seq_id));
    const size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), seq_id);
    buf.resize(n);
    return buf;
}

struct test_model {
    llama_model * model = nullptr;

    int n_embd  = 0;
    int n_layer = 0;

    static constexpr int n_seq_max = 4;

    llama_context * new_ctx(bool cvec_seq = true) const {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = 256*n_seq_max;
        cparams.n_batch         = 128;
        cparams.n_ubatch        = 16; // force several ubatches, some of them mixing sequences
        cparams.n_seq_max       = n_seq_max;
        cparams.n_threads       = 4;
        cparams.n_threads_batch = 4;
        cparams.no_perf         = true;
        cparams.cvec_seq        = cvec_seq;
        return llama_init_from_model(model, cparams);
    }

    // control vector data for layers [1, n_layer], same layout as common_control_vector_load
    std::vector<float> random_cvec(uint32_t seed, float stdev) const {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, stdev);
        std::vector<float> v((size_t) n_embd*n_layer);
        for (auto & x : v) {
            x = dist(rng);
        }
        return v;
    }

    std::vector<llama_token> random_tokens(uint32_t seed, int n) const {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::mt19937 rng(seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        std::vector<llama_token> t(n);
        for (auto & x : t) {
            x = dist(rng);
        }
        return t;
    }
};

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "    FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

// returns the number of failures, -1 if the model cannot run the stock path
static int test_model_file(const std::string & path, bool require_effect) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    test_model tm;
    tm.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!tm.model) {
        fprintf(stderr, "%s: failed to load model\n", path.c_str());
        return -1;
    }
    // non-causal models (diffusion, embeddings) need the whole prompt in one ubatch, not covered here
    if (llama_model_has_encoder(tm.model) || llama_model_is_diffusion(tm.model) || llama_vocab_n_tokens(llama_model_get_vocab(tm.model)) <= 0) {
        llama_model_free(tm.model);
        return -1;
    }

    tm.n_embd  = llama_model_n_embd(tm.model);
    tm.n_layer = llama_model_n_layer(tm.model);

    const auto p0 = tm.random_tokens(1, 24);
    const auto p1 = tm.random_tokens(2, 17);

    const auto v0 = tm.random_cvec(10, 1.0f);
    const auto v1 = tm.random_cvec(11, 1.0f);

    const int il_start = 1;
    const int il_end   = tm.n_layer;

    // reference: stock context without any control vector
    std::vector<float> ref_all;
    std::vector<float> ref_last;
    {
        llama_context * ctx = tm.new_ctx(false);
        if (ctx && !llama_get_causal_attn(ctx)) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (!ctx) {
            llama_model_free(tm.model);
            return -1;
        }
        bool ok = decode(ctx, {{0, p0}}, true, ref_all);
        llama_memory_clear(llama_get_memory(ctx), true);
        ok = ok && decode(ctx, {{0, p0}}, false, ref_last);
        llama_free(ctx);
        if (!ok) {
            llama_model_free(tm.model);
            return -1;
        }
    }

    int n_fail = 0;

    fprintf(stderr, "  %s (n_layer = %d, n_embd = %d)\n", std::filesystem::path(path).filename().string().c_str(), tm.n_layer, tm.n_embd);

    // 1. zero table
    {
        llama_context * ctx = tm.new_ctx();
        CHECK(llama_set_adapter_cvec_seq(ctx, 0, nullptr, 0, tm.n_embd, il_start, il_end) == 0, "clear seq 0");
        // a vector on another sequence must not leak into seq 0
        CHECK(llama_set_adapter_cvec_seq(ctx, 1, v1.data(), v1.size(), tm.n_embd, il_start, il_end) == 0, "set seq 1");

        std::vector<float> out_all, out_last;
        CHECK(decode(ctx, {{0, p0}}, true, out_all), "decode");
        llama_memory_clear(llama_get_memory(ctx), true);
        CHECK(decode(ctx, {{0, p0}}, false, out_last), "decode");
        llama_free(ctx);

        CHECK(out_all  == ref_all,  "zero row: logits (all) differ from stock, nmse = %g", nmse(ref_all, out_all));
        CHECK(out_last == ref_last, "zero row: logits (last) differ from stock, nmse = %g", nmse(ref_last, out_last));
        fprintf(stderr, "    [1] zero row == stock: %s\n", out_all == ref_all && out_last == ref_last ? "ok (bitwise)" : "FAIL");

        llama_context * ctx_off = tm.new_ctx(false);
        CHECK(llama_set_adapter_cvec_seq(ctx_off, 0, v0.data(), v0.size(), tm.n_embd, il_start, il_end) != 0, "accepted without cvec_seq");
        llama_free(ctx_off);
    }

    // 2. per-seq vs global, full and partial layer range, all and last-token outputs
    struct range { int s, e; };
    std::vector<range> ranges = {{il_start, il_end}};
    if (tm.n_layer >= 4) {
        ranges.push_back({2, tm.n_layer - 2});
    }
    bool has_effect = false;
    for (const auto & r : ranges) {
        for (bool all : {true, false}) {
            std::vector<float> out_glb, out_seq;

            llama_context * ctx_glb = tm.new_ctx(false);
            CHECK(llama_set_adapter_cvec(ctx_glb, v0.data(), v0.size(), tm.n_embd, r.s, r.e) == 0, "global cvec");
            CHECK(decode(ctx_glb, {{0, p0}}, all, out_glb), "decode");
            llama_free(ctx_glb);

            llama_context * ctx_seq = tm.new_ctx();
            CHECK(llama_set_adapter_cvec_seq(ctx_seq, 0, v0.data(), v0.size(), tm.n_embd, r.s, r.e) == 0, "seq cvec");
            CHECK(decode(ctx_seq, {{0, p0}}, all, out_seq), "decode");
            llama_free(ctx_seq);

            const double err = nmse(out_glb, out_seq);
            const double eff = nmse(all ? ref_all : ref_last, out_seq);
            has_effect |= eff > 1e-6;
            CHECK(out_glb == out_seq, "per-seq vs global [%d, %d] (%s): not bitwise equal, nmse = %g", r.s, r.e, all ? "all" : "last", err);
            fprintf(stderr, "    [2] per-seq == global, layers [%d, %d], outputs %-4s: %s (effect vs stock: nmse = %.3g)\n",
                    r.s, r.e, all ? "all" : "last", out_glb == out_seq ? "ok (bitwise)" : "FAIL", eff);
        }
    }
    if (require_effect) {
        CHECK(has_effect, "control vector has no effect on the logits");
    } else if (!has_effect) {
        fprintf(stderr, "    note: control vector has no effect on this model (build_cvec not used?)\n");
    }

    // 3. two sequences with different vectors in one batch vs each alone
    for (bool all : {true, false}) {
        std::vector<float> alone0, alone1, both;
        {
            llama_context * ctx = tm.new_ctx();
            llama_set_adapter_cvec_seq(ctx, 0, v0.data(), v0.size(), tm.n_embd, il_start, il_end);
            CHECK(decode(ctx, {{0, p0}}, all, alone0), "decode");
            llama_free(ctx);
        }
        {
            llama_context * ctx = tm.new_ctx();
            // use seq 2 to make sure the row index is not just the batch position
            llama_set_adapter_cvec_seq(ctx, 2, v1.data(), v1.size(), tm.n_embd, il_start, il_end);
            CHECK(decode(ctx, {{2, p1}}, all, alone1), "decode");
            llama_free(ctx);
        }
        {
            llama_context * ctx = tm.new_ctx();
            llama_set_adapter_cvec_seq(ctx, 0, v0.data(), v0.size(), tm.n_embd, il_start, il_end);
            llama_set_adapter_cvec_seq(ctx, 2, v1.data(), v1.size(), tm.n_embd, il_start, il_end);
            CHECK(decode(ctx, {{0, p0}, {2, p1}}, all, both), "decode");
            llama_free(ctx);
        }
        const size_t n0 = alone0.size();
        const std::vector<float> both0(both.begin(), both.begin() + std::min(n0, both.size()));
        const std::vector<float> both1(both.begin() + std::min(n0, both.size()), both.end());
        const double e0 = nmse(alone0, both0);
        const double e1 = nmse(alone1, both1);
        CHECK(e0 < 1e-6 && e1 < 1e-6, "batched vs alone (%s): nmse = %g, %g", all ? "all" : "last", e0, e1);
        fprintf(stderr, "    [3] 2 seqs batched == alone, outputs %-4s: nmse = %.3g, %.3g\n", all ? "all" : "last", e0, e1);
    }

    // 3b. isolation: same batch, only the row of the other sequence changes -> bitwise identical logits for seq 0
    for (bool all : {true, false}) {
        std::vector<float> out_a, out_b;
        for (int k = 0; k < 2; ++k) {
            llama_context * ctx = tm.new_ctx();
            llama_set_adapter_cvec_seq(ctx, 0, v0.data(), v0.size(), tm.n_embd, il_start, il_end);
            if (k == 1) {
                llama_set_adapter_cvec_seq(ctx, 2, v1.data(), v1.size(), tm.n_embd, il_start, il_end);
            }
            CHECK(decode(ctx, {{0, p0}, {2, p1}}, all, k == 0 ? out_a : out_b), "decode");
            llama_free(ctx);
        }
        const size_t n0 = all ? p0.size() : 1;
        const size_t n_vocab = out_a.size() / (all ? p0.size() + p1.size() : 2);
        const bool same0 = std::equal(out_a.begin(), out_a.begin() + n0*n_vocab, out_b.begin());
        const bool same1 = std::equal(out_a.begin() + n0*n_vocab, out_a.end(), out_b.begin() + n0*n_vocab);
        CHECK(same0, "isolation (%s): seq 0 logits changed when only the row of seq 2 changed", all ? "all" : "last");
        if (require_effect) {
            CHECK(!same1, "isolation (%s): seq 2 logits did not change", all ? "all" : "last");
        }
        fprintf(stderr, "    [3b] other seq row flipped, outputs %-4s: seq 0 %s, seq 2 %s\n", all ? "all" : "last",
                same0 ? "unchanged (bitwise)" : "CHANGED", same1 ? "unchanged" : "changed");
    }

    // 4. seq 0 prefills unsteered while seq 1 is steered in the same batch -> seq 0 state is identical to a stock run
    //    then steer seq 0 for a few "generated" tokens and check that its state changes
    {
        llama_context * ctx_ref = tm.new_ctx(false);
        llama_context * ctx     = tm.new_ctx();

        std::vector<float> tmp;
        CHECK(decode(ctx_ref, {{1, p1}}, false, tmp), "decode");
        CHECK(decode(ctx,     {{1, p1}}, false, tmp), "decode");

        llama_set_adapter_cvec_seq(ctx, 0, nullptr, 0, tm.n_embd, il_start, il_end);
        llama_set_adapter_cvec_seq(ctx, 1, v1.data(), v1.size(), tm.n_embd, il_start, il_end);

        const auto g1 = tm.random_tokens(3, 1);
        CHECK(decode(ctx_ref, {{0, p0}, {1, g1}}, false, tmp), "decode");
        CHECK(decode(ctx,     {{0, p0}, {1, g1}}, false, tmp), "decode");

        const auto s0_ref = seq_state(ctx_ref, 0);
        const auto s0     = seq_state(ctx,     0);
        const auto s1_ref = seq_state(ctx_ref, 1);
        const auto s1     = seq_state(ctx,     1);

        CHECK(!s0.empty() && s0 == s0_ref, "unsteered prefill state differs from stock (size %zu vs %zu)", s0.size(), s0_ref.size());
        // the vector of layer il reaches the KV of the later layers only
        if (require_effect && tm.n_layer >= 3) {
            CHECK(s1 != s1_ref, "steered sequence state is identical to stock");
        }
        fprintf(stderr, "    [4] prefill state with steering on another seq == stock: %s (%zu bytes)\n", s0 == s0_ref ? "ok (bitwise)" : "FAIL", s0.size());

        // 5. switching rows between decodes must not re-reserve
        const int n_reserve0 = g_n_reserve.load();
        for (int i = 0; i < 4; ++i) {
            llama_set_adapter_cvec_seq(ctx, 0, (i % 2 ? v1 : v0).data(), v0.size(), tm.n_embd, il_start, il_end);
            llama_set_adapter_cvec_seq(ctx, 1, i % 2 ? nullptr : v1.data(), v1.size(), tm.n_embd, il_start, il_end);
            CHECK(decode(ctx, {{0, tm.random_tokens(20 + i, 1)}, {1, tm.random_tokens(30 + i, 1)}}, false, tmp), "decode");
        }
        const int n_reserve = g_n_reserve.load() - n_reserve0;
        CHECK(n_reserve == 0, "table updates triggered %d re-reserve(s)", n_reserve);
        fprintf(stderr, "    [5] table updates between decodes: %d re-reserve(s)\n", n_reserve);

        CHECK(llama_set_adapter_cvec_seq(ctx, test_model::n_seq_max, v0.data(), v0.size(), tm.n_embd, il_start, il_end) != 0, "out of range seq_id accepted");
        CHECK(llama_set_adapter_cvec_seq(ctx, 0, v0.data(), v0.size(), tm.n_embd + 1, il_start, il_end) != 0, "wrong n_embd accepted");

        llama_free(ctx_ref);
        llama_free(ctx);
    }

    llama_model_free(tm.model);

    return n_fail;
}

int main(int argc, char ** argv) {
    std::vector<std::string> models;
    std::string models_dir;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            models.push_back(argv[++i]);
        } else if (arg == "--models" && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "-v") {
            g_verbose = true;
        } else {
            fprintf(stderr, "usage: %s [-m model.gguf]... [--models DIR] [-v]\n", argv[0]);
            return 1;
        }
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    // models given with -m must run, models found in a directory may be skipped
    std::vector<std::pair<std::string, bool>> todo;
    for (const auto & m : models) {
        todo.push_back({m, true});
    }
    if (!models_dir.empty()) {
        std::vector<std::string> found;
        for (const auto & e : std::filesystem::directory_iterator(models_dir)) {
            if (e.path().extension() == ".gguf") {
                found.push_back(e.path().string());
            }
        }
        std::sort(found.begin(), found.end());
        for (const auto & m : found) {
            todo.push_back({m, false});
        }
    }
    if (todo.empty()) {
        fprintf(stderr, "no models given\n");
        return 1;
    }

    int n_fail    = 0;
    int n_skip    = 0;
    int n_tested  = 0;
    for (const auto & [path, required] : todo) {
        const int res = test_model_file(path, required);
        if (res < 0) {
            fprintf(stderr, "  %s: %s\n", path.c_str(), required ? "FAIL: cannot run the stock path" : "skipped (cannot run the stock path)");
            n_fail += required;
            n_skip += !required;
            continue;
        }
        n_tested++;
        n_fail += res;
    }

    llama_backend_free();

    fprintf(stderr, "%s: %d model(s) tested, %d skipped, %d failure(s)\n", __func__, n_tested, n_skip, n_fail);

    return n_fail == 0 ? 0 : 1;
}
