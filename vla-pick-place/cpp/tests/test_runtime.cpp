// Tests for the C++ runtime: numerical parity with PyTorch and input validation.
// usage: test_runtime <weights.bin> <fixtures.bin>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/fixtures.hpp"
#include "vla/policy.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& name) {
  std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
  if (!ok) ++failures;
}

bool throws(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

std::string truncated_copy(const std::string& src, size_t keep) {
  std::ifstream in(src, std::ios::binary);
  std::vector<char> buf(keep);
  in.read(buf.data(), static_cast<std::streamsize>(keep));
  const std::string dst = "truncated_weights.bin";
  std::ofstream(dst, std::ios::binary).write(buf.data(), static_cast<std::streamsize>(in.gcount()));
  return dst;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <weights.bin> <fixtures.bin>\n", argv[0]);
    return 2;
  }
  vla::Policy policy(argv[1]);
  const auto& cfg = policy.config();
  const auto fx = vla::load_fixtures(argv[2], cfg);
  check(!fx.empty(), "fixtures loaded");

  float max_err = 0.f;
  for (const auto& f : fx) {
    const auto y = policy.predict(f.image.data(), f.tokens.data(), f.proprio.data());
    for (size_t i = 0; i < y.size(); ++i) max_err = std::max(max_err, std::fabs(y[i] - f.expected[i]));
  }
  std::printf("max abs error vs PyTorch: %.3g\n", max_err);
  check(max_err < 1e-4f, "outputs match PyTorch within 1e-4");

  const auto& f0 = fx[0];
  const auto a = policy.predict(f0.image.data(), f0.tokens.data(), f0.proprio.data());
  const auto b = policy.predict(f0.image.data(), f0.tokens.data(), f0.proprio.data());
  check(a == b, "repeated calls are bit-identical");

  std::vector<int32_t> bad(f0.tokens);
  bad[0] = cfg.vocab + 5;
  check(throws([&] { policy.predict(f0.image.data(), bad.data(), f0.proprio.data()); }),
        "out-of-vocabulary token id is rejected");
  check(throws([] { vla::Policy("does_not_exist.bin"); }), "missing weights file is rejected");
  check(throws([&] { vla::Policy(truncated_copy(argv[1], 1000)); }), "truncated weights are rejected");

  std::printf("%d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
