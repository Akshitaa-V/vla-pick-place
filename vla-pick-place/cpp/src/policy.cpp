#include "vla/policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace vla {
namespace {

constexpr char kMagic[4] = {'V', 'L', 'A', 'P'};
constexpr uint32_t kVersion = 4;
constexpr float kLnEps = 1e-5f;

class Reader {
 public:
  explicit Reader(const std::string& path) : in_(path, std::ios::binary) {
    if (!in_) throw std::runtime_error("cannot open weights file: " + path);
  }
  template <typename T>
  T scalar() {
    T v{};
    read(&v, sizeof(T));
    return v;
  }
  std::vector<float> tensor(size_t expected) {
    const uint32_t n = scalar<uint32_t>();
    if (n != expected)
      throw std::runtime_error("weights file: tensor has " + std::to_string(n) +
                               " values, expected " + std::to_string(expected));
    std::vector<float> v(n);
    read(v.data(), n * sizeof(float));
    return v;
  }
  void read(void* dst, size_t bytes) {
    in_.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (in_.gcount() != static_cast<std::streamsize>(bytes))
      throw std::runtime_error("weights file is truncated");
  }
  bool at_end() { return in_.peek() == std::char_traits<char>::eof(); }

 private:
  std::ifstream in_;
};

// Dot product with 8 independent accumulators so the compiler can vectorise it
// without -ffast-math.
inline float dot(const float* a, const float* b, int n) {
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int i = 0;
  for (; i + 8 <= n; i += 8)
    for (int k = 0; k < 8; ++k) acc[k] += a[i + k] * b[i + k];
  float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)); }

// PyTorch stores a weight as (out, in); the kernel below wants (in, out).
std::vector<float> transpose(const std::vector<float>& w, int out, int in) {
  std::vector<float> t(w.size());
  for (int o = 0; o < out; ++o)
    for (int i = 0; i < in; ++i) t[static_cast<size_t>(i) * out + o] = w[static_cast<size_t>(o) * in + i];
  return t;
}

// y[rows, out] = x[rows, in] * wt[in, out] + b. Four rows share each pass over a weight
// row, and the inner loop is a contiguous multiply-add over outputs, which compilers
// turn into packed FMA instructions.
void gemm(const float* x, int rows, int in, const float* wt, const float* b, int out, float* y) {
  const int blocks = (rows + 3) / 4;
  // Only spread work across threads when there is enough of it to pay for the hand-off.
  const bool big = static_cast<long>(rows) * in * out > 200000L;
#pragma omp parallel for schedule(static) if (big)
  for (int blk = 0; blk < blocks; ++blk) {
    const int r0 = blk * 4, nr = std::min(4, rows - r0);
    float* yr[4];
    const float* xr[4];
    for (int r = 0; r < 4; ++r) {
      const int rr = r0 + std::min(r, nr - 1);   // duplicate the last row in a partial block
      yr[r] = y + static_cast<size_t>(rr) * out;
      xr[r] = x + static_cast<size_t>(rr) * in;
    }
    float acc[4][512];
    for (int r = 0; r < 4; ++r) std::copy_n(b, out, acc[r]);
    for (int i = 0; i < in; ++i) {
      const float* w = wt + static_cast<size_t>(i) * out;
      const float x0 = xr[0][i], x1 = xr[1][i], x2 = xr[2][i], x3 = xr[3][i];
      for (int o = 0; o < out; ++o) {
        const float wo = w[o];
        acc[0][o] += x0 * wo;
        acc[1][o] += x1 * wo;
        acc[2][o] += x2 * wo;
        acc[3][o] += x3 * wo;
      }
    }
    for (int r = 0; r < nr; ++r) std::copy_n(acc[r], out, yr[r]);
  }
}

constexpr int kMaxGemmOut = 512;

size_t hw_size_of(const Config& k) { return static_cast<size_t>(k.img_h) * k.img_w; }

}  // namespace

Policy::Policy(const std::string& path) {
  Reader r(path);
  char magic[4];
  r.read(magic, 4);
  if (std::memcmp(magic, kMagic, 4) != 0) throw std::runtime_error("not a VLAP weights file");
  if (r.scalar<uint32_t>() != kVersion) throw std::runtime_error("unsupported weights version");
  int32_t c[15];
  r.read(c, sizeof(c));
  cfg_ = Config{c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14]};
  const Config& k = cfg_;
  if (k.dim <= 0 || k.heads <= 0 || k.dim % k.heads != 0 || k.patch != 8 || k.img_h % k.patch ||
      k.img_w % k.patch || k.stem1 <= 0 || k.stem2 <= 0 || k.n_keypoints <= 0)
    throw std::runtime_error("weights file: invalid model configuration");

  const size_t d = k.dim, L = k.seq_len();
  conv1_w_ = r.tensor(static_cast<size_t>(k.stem1) * 5 * 5 * 5);
  conv1_b_ = r.tensor(k.stem1);
  conv2_w_ = r.tensor(static_cast<size_t>(k.stem2) * k.stem1 * 3 * 3);
  conv2_b_ = r.tensor(k.stem2);
  conv3_w_ = r.tensor(d * k.stem2 * 3 * 3);
  conv3_b_ = r.tensor(d);
  film_w_ = r.tensor(2 * static_cast<size_t>(k.stem2) * d);
  film_b_ = r.tensor(2 * static_cast<size_t>(k.stem2));
  kp_conv_w_ = r.tensor(static_cast<size_t>(k.n_keypoints) * k.stem2 * 3 * 3);
  kp_conv_b_ = r.tensor(k.n_keypoints);
  kp_scale_ = std::exp(r.tensor(1)[0]);
  kp_embed_w_ = r.tensor(d * 2 * k.n_keypoints);
  kp_embed_b_ = r.tensor(d);
  word_emb_ = r.tensor(static_cast<size_t>(k.vocab) * d);
  proprio_w_ = r.tensor(d * k.proprio_dim);
  proprio_b_ = r.tensor(d);
  action_query_ = r.tensor(d);
  pos_ = r.tensor(L * d);
  layers_.resize(k.depth);
  for (auto& w : layers_) {
    w.ln1_w = r.tensor(d);
    w.ln1_b = r.tensor(d);
    w.qkv_w = r.tensor(3 * d * d);
    w.qkv_b = r.tensor(3 * d);
    w.proj_w = r.tensor(d * d);
    w.proj_b = r.tensor(d);
    w.ln2_w = r.tensor(d);
    w.ln2_b = r.tensor(d);
    w.fc1_w = r.tensor(static_cast<size_t>(k.mlp_dim) * d);
    w.fc1_b = r.tensor(k.mlp_dim);
    w.fc2_w = r.tensor(d * k.mlp_dim);
    w.fc2_b = r.tensor(d);
  }
  ln_out_w_ = r.tensor(d);
  ln_out_b_ = r.tensor(d);
  head_w_ = r.tensor(static_cast<size_t>(k.output_size()) * d);
  head_b_ = r.tensor(k.output_size());
  if (!r.at_end()) throw std::runtime_error("weights file has trailing data");
  if (std::max({3 * k.dim, k.mlp_dim, k.output_size(), 2 * k.stem2}) > kMaxGemmOut)
    throw std::runtime_error("model is wider than the runtime supports");

  // Store every weight matrix transposed to (in, out) for the gemm kernel.
  conv1_w_ = transpose(conv1_w_, k.stem1, 5 * 5 * 5);
  conv2_w_ = transpose(conv2_w_, k.stem2, k.stem1 * 9);
  conv3_w_ = transpose(conv3_w_, k.dim, k.stem2 * 9);
  kp_conv_w_ = transpose(kp_conv_w_, k.n_keypoints, k.stem2 * 9);
  film_w_ = transpose(film_w_, 2 * k.stem2, k.dim);
  kp_embed_w_ = transpose(kp_embed_w_, k.dim, 2 * k.n_keypoints);
  proprio_w_ = transpose(proprio_w_, k.dim, k.proprio_dim);
  for (auto& w : layers_) {
    w.qkv_w = transpose(w.qkv_w, 3 * k.dim, k.dim);
    w.proj_w = transpose(w.proj_w, k.dim, k.dim);
    w.fc1_w = transpose(w.fc1_w, k.mlp_dim, k.dim);
    w.fc2_w = transpose(w.fc2_w, k.dim, k.mlp_dim);
  }
  head_w_ = transpose(head_w_, k.output_size(), k.dim);
  pix_.resize((hw_size_of(k) / 4) * std::max(k.stem1, k.dim));
  const size_t hd = k.dim / k.heads;
  const size_t heads = k.heads;
  q_.resize(heads * L * hd);
  kt_.resize(heads * hd * L);
  v_.resize(heads * L * hd);
  scores_.resize(heads * L * L);
  head_out_.resize(heads * L * hd);

  x_.resize(L * d);
  norm_.resize(L * d);
  qkv_.resize(L * 3 * d);
  att_.resize(L * d);
  tmp_.resize(L * d);
  hidden_.resize(L * static_cast<size_t>(k.mlp_dim));
  const size_t hw = static_cast<size_t>(k.img_h) * k.img_w;
  input_.resize(5 * hw);
  feat1_.resize(k.stem1 * hw / 4);
  feat2_.resize(k.stem2 * hw / 16);
  feat3_.resize(d * hw / 64);
  filmed_.resize(k.stem2 * hw / 16);
  heat_.resize(k.n_keypoints * hw / 16);
  keypoints_.resize(2 * k.n_keypoints);
  cols_.resize((hw / 4) * std::max({5 * 5 * 5, k.stem1 * 9}) + (hw / 16) * k.stem2 * 9);
  key_mask_.assign(L, 1);
}

void Policy::linear(const float* x, int rows, int in, const std::vector<float>& wt,
                    const std::vector<float>& b, int out, float* y) const {
  gemm(x, rows, in, wt.data(), b.data(), out, y);
}

void Policy::layer_norm(const float* x, int rows, const std::vector<float>& w,
                        const std::vector<float>& b, float* y) const {
  const int d = cfg_.dim;
  for (int t = 0; t < rows; ++t) {
    const float* xt = x + static_cast<size_t>(t) * d;
    float* yt = y + static_cast<size_t>(t) * d;
    float mean = 0.f;
    for (int i = 0; i < d; ++i) mean += xt[i];
    mean /= d;
    float var = 0.f;
    for (int i = 0; i < d; ++i) var += (xt[i] - mean) * (xt[i] - mean);
    const float inv = 1.0f / std::sqrt(var / d + kLnEps);
    for (int i = 0; i < d; ++i) yt[i] = (xt[i] - mean) * inv * w[i] + b[i];
  }
}

void Policy::attention(const float* qkv, const std::vector<uint8_t>& key_mask, float* out, int n_queries) {
  const int L = cfg_.seq_len(), d = cfg_.dim, H = cfg_.heads, hd = d / H;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const size_t stride = 3 * static_cast<size_t>(d);
  // Masked keys are dropped up front, so softmax runs over the kept keys only.
  std::vector<int> keys;
  keys.reserve(L);
  for (int j = 0; j < L; ++j)
    if (key_mask[j]) keys.push_back(j);
  const int nk = static_cast<int>(keys.size());
  const std::vector<float> zeros(std::max(nk, hd), 0.f);
  const size_t L2 = static_cast<size_t>(L) * L, Lh = static_cast<size_t>(L) * hd;
  // Heads are independent: each one gets its own slice of the scratch buffers.
#pragma omp parallel for schedule(static) if (n_queries > 1)
  for (int h = 0; h < H; ++h) {
    float* q = q_.data() + h * Lh;
    float* kt = kt_.data() + h * Lh;
    float* v = v_.data() + h * Lh;
    float* scores = scores_.data() + h * L2;
    float* head_out = head_out_.data() + h * Lh;
    // Pack this head's queries (n_queries, hd), keys transposed (hd, nk) and values (nk, hd).
    for (int i = 0; i < n_queries; ++i)
      for (int e = 0; e < hd; ++e) q[static_cast<size_t>(i) * hd + e] = qkv[i * stride + h * hd + e] * scale;
    for (int jj = 0; jj < nk; ++jj) {
      const float* kv = qkv + keys[jj] * stride + d + h * hd;
      for (int e = 0; e < hd; ++e) {
        kt[static_cast<size_t>(e) * nk + jj] = kv[e];
        v[static_cast<size_t>(jj) * hd + e] = kv[d + e];
      }
    }
    gemm(q, n_queries, hd, kt, zeros.data(), nk, scores);   // (n_queries, nk)
    for (int i = 0; i < n_queries; ++i) {
      float* sr = scores + static_cast<size_t>(i) * nk;
      const float mx = *std::max_element(sr, sr + nk);
      float sum = 0.f;
      for (int jj = 0; jj < nk; ++jj) {
        sr[jj] = std::exp(sr[jj] - mx);
        sum += sr[jj];
      }
      const float inv = 1.0f / sum;
      for (int jj = 0; jj < nk; ++jj) sr[jj] *= inv;
    }
    gemm(scores, n_queries, nk, v, zeros.data(), hd, head_out);   // (n_queries, hd)
    for (int i = 0; i < n_queries; ++i)
      std::copy_n(head_out + static_cast<size_t>(i) * hd, hd, out + static_cast<size_t>(i) * d + h * hd);
  }
}

void Policy::conv2d(const float* in, int c_in, int h, int w, const std::vector<float>& weight,
                    const std::vector<float>& bias, int c_out, int k, int stride, int pad, float* out,
                    int* h_out, int* w_out) {
  const int ho = (h + 2 * pad - k) / stride + 1, wo = (w + 2 * pad - k) / stride + 1;
  const int ck = c_in * k * k;
  // im2col: one contiguous row of (c, ky, kx) values per output pixel, zero outside the image.
  for (int oy = 0; oy < ho; ++oy)
    for (int ox = 0; ox < wo; ++ox) {
      float* col = cols_.data() + static_cast<size_t>(oy * wo + ox) * ck;
      for (int c = 0; c < c_in; ++c)
        for (int ky = 0; ky < k; ++ky)
          for (int kx = 0; kx < k; ++kx) {
            const int y = oy * stride - pad + ky, x = ox * stride - pad + kx;
            *col++ = (y >= 0 && y < h && x >= 0 && x < w) ? in[(static_cast<size_t>(c) * h + y) * w + x] : 0.f;
          }
    }
  const int n = ho * wo;
  gemm(cols_.data(), n, ck, weight.data(), bias.data(), c_out, pix_.data());   // (pixels, c_out)
  for (int p = 0; p < n; ++p)
    for (int o = 0; o < c_out; ++o) out[static_cast<size_t>(o) * n + p] = pix_[static_cast<size_t>(p) * c_out + o];
  *h_out = ho;
  *w_out = wo;
}

std::vector<float> Policy::predict(const uint8_t* image, const int32_t* tokens, const float* proprio) {
  const Config& k = cfg_;
  const int d = k.dim, L = k.seq_len(), H = k.img_h, W = k.img_w, np = k.n_patches();

  // Stem input: RGB scaled to [-1, 1] plus x and y coordinate channels in [-1, 1], (C, H, W).
  const size_t hw = static_cast<size_t>(H) * W;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t i = static_cast<size_t>(y) * W + x;
      for (int c = 0; c < 3; ++c) input_[c * hw + i] = image[i * 3 + c] / 127.5f - 1.0f;
      input_[3 * hw + i] = W > 1 ? -1.0f + 2.0f * x / (W - 1) : 0.f;
      input_[4 * hw + i] = H > 1 ? -1.0f + 2.0f * y / (H - 1) : 0.f;
    }
  int h1, w1, h2, w2, h3, w3;
  conv2d(input_.data(), 5, H, W, conv1_w_, conv1_b_, k.stem1, 5, 2, 2, feat1_.data(), &h1, &w1);
  for (auto& v : feat1_) v = gelu(v);
  conv2d(feat1_.data(), k.stem1, h1, w1, conv2_w_, conv2_b_, k.stem2, 3, 2, 1, feat2_.data(), &h2, &w2);
  for (auto& v : feat2_) v = gelu(v);
  conv2d(feat2_.data(), k.stem2, h2, w2, conv3_w_, conv3_b_, d, 3, 2, 1, feat3_.data(), &h3, &w3);
  if (h3 * w3 != np) throw std::logic_error("stem output does not match the token grid");

  for (int t = 0; t < k.max_tokens; ++t)
    if (tokens[t] < 0 || tokens[t] >= k.vocab) throw std::out_of_range("token id out of vocabulary range");

  // FiLM: per-channel scale and shift from the mean embedding of the non-padding words.
  std::vector<float> lang(d, 0.f), film(2 * k.stem2);
  int n_words = 0;
  for (int t = 0; t < k.max_tokens; ++t) {
    if (tokens[t] == 0) continue;
    const float* e = word_emb_.data() + static_cast<size_t>(tokens[t]) * d;
    for (int i = 0; i < d; ++i) lang[i] += e[i];
    ++n_words;
  }
  for (auto& v : lang) v /= static_cast<float>(std::max(n_words, 1));
  linear(lang.data(), 1, d, film_w_, film_b_, 2 * k.stem2, film.data());
  const int n2 = h2 * w2;
  for (int c = 0; c < k.stem2; ++c)
    for (int p = 0; p < n2; ++p) {
      const size_t i = static_cast<size_t>(c) * n2 + p;
      filmed_[i] = gelu(feat2_[i] * (1.f + film[c]) + film[k.stem2 + c]);
    }

  // Keypoint heatmaps -> spatial softmax -> expected (u, v) on a [-1, 1] grid.
  int hk, wk;
  conv2d(filmed_.data(), k.stem2, h2, w2, kp_conv_w_, kp_conv_b_, k.n_keypoints, 3, 1, 1, heat_.data(), &hk, &wk);
  for (int j = 0; j < k.n_keypoints; ++j) {
    float* hm = heat_.data() + static_cast<size_t>(j) * n2;
    for (int p = 0; p < n2; ++p) hm[p] *= kp_scale_;
    const float mx = *std::max_element(hm, hm + n2);
    double sum = 0.0, su = 0.0, sv = 0.0;
    for (int y = 0; y < hk; ++y)
      for (int x = 0; x < wk; ++x) {
        const double e = std::exp(static_cast<double>(hm[y * wk + x] - mx));
        sum += e;
        su += e * (wk > 1 ? -1.0 + 2.0 * x / (wk - 1) : 0.0);
        sv += e * (hk > 1 ? -1.0 + 2.0 * y / (hk - 1) : 0.0);
      }
    keypoints_[2 * j] = static_cast<float>(su / sum);
    keypoints_[2 * j + 1] = static_cast<float>(sv / sum);
  }

  // Token sequence: [action query, proprio, keypoints, words..., scene...] + positional embedding.
  std::copy(action_query_.begin(), action_query_.end(), x_.begin());
  linear(proprio, 1, k.proprio_dim, proprio_w_, proprio_b_, d, x_.data() + d);
  linear(keypoints_.data(), 1, 2 * k.n_keypoints, kp_embed_w_, kp_embed_b_, d, x_.data() + 2 * d);
  for (int t = 0; t < k.max_tokens; ++t) {
    std::copy_n(word_emb_.data() + static_cast<size_t>(tokens[t]) * d, d, x_.data() + (3 + t) * d);
    key_mask_[3 + t] = tokens[t] != 0;
  }
  float* img_tokens = x_.data() + static_cast<size_t>(3 + k.max_tokens) * d;
  for (int t = 0; t < np; ++t)
    for (int c = 0; c < d; ++c) img_tokens[static_cast<size_t>(t) * d + c] = feat3_[static_cast<size_t>(c) * np + t];
  for (size_t i = 0; i < x_.size(); ++i) x_[i] += pos_[i];

  for (size_t li = 0; li < layers_.size(); ++li) {
    const auto& w = layers_[li];
    // In the last layer only the ACTION token (row 0) is read out, so queries, the
    // attention output and the MLP are computed for that row alone. Keys and values
    // still come from every token, so the result is identical to the full pass.
    const int rows = li + 1 == layers_.size() ? 1 : L;
    layer_norm(x_.data(), L, w.ln1_w, w.ln1_b, norm_.data());
    linear(norm_.data(), L, d, w.qkv_w, w.qkv_b, 3 * d, qkv_.data());
    attention(qkv_.data(), key_mask_, att_.data(), rows);
    linear(att_.data(), rows, d, w.proj_w, w.proj_b, d, tmp_.data());
    for (size_t i = 0; i < static_cast<size_t>(rows) * d; ++i) x_[i] += tmp_[i];

    layer_norm(x_.data(), rows, w.ln2_w, w.ln2_b, norm_.data());
    linear(norm_.data(), rows, d, w.fc1_w, w.fc1_b, k.mlp_dim, hidden_.data());
    for (size_t i = 0; i < static_cast<size_t>(rows) * k.mlp_dim; ++i) hidden_[i] = gelu(hidden_[i]);
    linear(hidden_.data(), rows, k.mlp_dim, w.fc2_w, w.fc2_b, d, tmp_.data());
    for (size_t i = 0; i < static_cast<size_t>(rows) * d; ++i) x_[i] += tmp_[i];
  }

  // Only the action token is read out.
  std::vector<float> h(d), out(k.output_size());
  layer_norm(x_.data(), 1, ln_out_w_, ln_out_b_, h.data());
  linear(h.data(), 1, d, head_w_, head_b_, k.output_size(), out.data());
  return out;
}

}  // namespace vla
