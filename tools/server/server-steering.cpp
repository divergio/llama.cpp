#include "server-steering.h"

#include "common.h"
#include "hash/hash.h"

#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

server_steering_vector server_steering_registry::load(const std::string & name, const std::string & path, int32_t n_embd, int32_t n_layer) {
    if (name.empty()) {
        throw std::invalid_argument("steering vector name must not be empty");
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::invalid_argument("failed to open steering vector file: " + path);
    }
    const std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    common_control_vector_data cvec = common_control_vector_load({ { 1.0f, path } }, n_layer);
    if (cvec.n_embd == -1) {
        throw std::invalid_argument("invalid steering vector file: " + path);
    }
    if (cvec.n_embd != n_embd) {
        throw std::invalid_argument("steering vector " + name + ": n_embd = " + std::to_string(cvec.n_embd) +
                " does not match the model (n_embd = " + std::to_string(n_embd) + ")");
    }

    server_steering_vector res;
    res.name   = name;
    res.path   = path;
    res.hash   = hash_sha256_hex(bytes.data(), bytes.size());
    res.n_embd = cvec.n_embd;
    res.data   = std::move(cvec.data);
    res.data.resize((size_t) n_embd*n_layer, 0.0f);

    return res;
}

bool server_steering_registry::add(server_steering_vector && vec) {
    auto ptr = std::make_shared<const server_steering_vector>(std::move(vec));

    std::lock_guard<std::mutex> lock(mtx);
    const bool replaced = vecs.count(ptr->name) > 0;
    vecs[ptr->name] = std::move(ptr);
    return replaced;
}

void server_steering_registry::validate(const std::map<std::string, float> & scales) const {
    std::lock_guard<std::mutex> lock(mtx);
    for (const auto & [name, scale] : scales) {
        if (vecs.find(name) == vecs.end()) {
            throw std::invalid_argument("unknown steering vector: '" + name + "'");
        }
        if (!std::isfinite(scale)) {
            throw std::invalid_argument("steering scale of '" + name + "' must be a finite number");
        }
    }
}

server_steering_mix server_steering_registry::mix(const std::map<std::string, float> & scales) const {
    server_steering_mix res;
    res.info = json::object();

    std::lock_guard<std::mutex> lock(mtx);
    for (const auto & [name, scale] : scales) {
        const auto it = vecs.find(name);
        if (it == vecs.end()) {
            continue;
        }
        const auto & vec = *it->second;

        res.info[name] = {
            {"scale", scale},
            {"hash",  vec.hash},
        };

        if (scale == 0.0f) {
            continue;
        }
        if (res.data.empty()) {
            res.data.resize(vec.data.size(), 0.0f);
        }
        GGML_ASSERT(res.data.size() == vec.data.size());
        for (size_t i = 0; i < vec.data.size(); ++i) {
            res.data[i] += scale*vec.data[i];
        }
    }

    return res;
}

json server_steering_registry::to_json() const {
    std::lock_guard<std::mutex> lock(mtx);
    json res = json::array();
    for (const auto & [name, vec] : vecs) {
        res.push_back({
            {"name", name},
            {"path", vec->path},
            {"hash", vec->hash},
        });
    }
    return res;
}

bool server_steering_registry::empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return vecs.empty();
}
