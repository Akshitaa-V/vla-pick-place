// Latency benchmark: vla_bench <weights.bin> <fixtures.bin> [iterations]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fixtures.hpp"
#include "vla/policy.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <weights.bin> <fixtures.bin> [iterations]\n", argv[0]);
    return 2;
  }
  const int iters = argc > 3 ? std::atoi(argv[3]) : 200;
  vla::Policy policy(argv[1]);
  const auto fx = vla::load_fixtures(argv[2], policy.config());
  if (fx.empty()) return 2;

  float max_err = 0.f;
  for (const auto& f : fx) {
    const auto y = policy.predict(f.image.data(), f.tokens.data(), f.proprio.data());
    for (size_t i = 0; i < y.size(); ++i) max_err = std::max(max_err, std::fabs(y[i] - f.expected[i]));
  }

  for (int i = 0; i < 10; ++i) policy.predict(fx[0].image.data(), fx[0].tokens.data(), fx[0].proprio.data());
  std::vector<double> ms(iters);
  for (int i = 0; i < iters; ++i) {
    const auto& f = fx[i % fx.size()];
    const auto t0 = std::chrono::steady_clock::now();
    policy.predict(f.image.data(), f.tokens.data(), f.proprio.data());
    ms[i] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  std::sort(ms.begin(), ms.end());
  std::printf("{\"fixtures\": %zu, \"max_abs_error\": %.3g, \"median_ms\": %.3f, \"p95_ms\": %.3f}\n",
              fx.size(), max_err, ms[iters / 2], ms[static_cast<size_t>(iters * 0.95)]);
  return 0;
}
