#pragma once

#include "server-common.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// named steering vectors for per-request activation steering, see STEERING.md
// the vectors use the control vector GGUF format of llama-cvector-generator and are kept in CPU memory

struct server_steering_vector {
    std::string name;
    std::string path;
    std::string hash; // sha256 of the file

    int32_t n_embd = 0;

    // layers [1, n_layer], layer 1 at offset 0, see common_control_vector_data
    std::vector<float> data;
};

using server_steering_vector_ptr = std::shared_ptr<const server_steering_vector>;

// the mix of a request: sum of scale * vector
struct server_steering_mix {
    std::vector<float> data; // empty: no steering

    json info; // echoed in the response: {"name": {"scale": s, "hash": h}, ...}

    bool empty() const { return data.empty(); }
};

class server_steering_registry {
public:
    // read and validate a vector file, does not change the registry. throws std::invalid_argument
    static server_steering_vector load(const std::string & name, const std::string & path, int32_t n_embd, int32_t n_layer);

    // add or replace a vector, returns true if a vector with the same name was replaced
    bool add(server_steering_vector && vec);

    // throws std::invalid_argument for unknown names or non-finite scales
    void validate(const std::map<std::string, float> & scales) const;

    // vectors that are no longer registered are skipped, scales of 0 are listed in the info but not mixed
    server_steering_mix mix(const std::map<std::string, float> & scales) const;

    json to_json() const;

    bool empty() const;

private:
    mutable std::mutex mtx;

    std::map<std::string, server_steering_vector_ptr> vecs;
};
