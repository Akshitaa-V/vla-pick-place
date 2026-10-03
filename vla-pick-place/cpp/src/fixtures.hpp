// Reads parity fixtures written by `python -m vla.export`.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vla/policy.hpp"

namespace vla {

struct Fixture {
  std::vector<uint8_t> image;
  std::vector<int32_t> tokens;
  std::vector<float> proprio;
  std::vector<float> expected;
};

inline std::vector<Fixture> load_fixtures(const std::string& path, const Config& cfg) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open fixtures: " + path);
  uint32_t n = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  std::vector<Fixture> out(n);
  for (auto& f : out) {
    f.image.resize(static_cast<size_t>(cfg.img_h) * cfg.img_w * 3);
    f.tokens.resize(cfg.max_tokens);
    f.proprio.resize(cfg.proprio_dim);
    f.expected.resize(cfg.output_size());
    in.read(reinterpret_cast<char*>(f.image.data()), f.image.size());
    in.read(reinterpret_cast<char*>(f.tokens.data()), f.tokens.size() * sizeof(int32_t));
    in.read(reinterpret_cast<char*>(f.proprio.data()), f.proprio.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(f.expected.data()), f.expected.size() * sizeof(float));
    if (!in) throw std::runtime_error("fixtures file is truncated");
  }
  return out;
}

}  // namespace vla
