// Dependency-free C++17 inference runtime for the VLA pick-and-place policy.
//
// Loads weights exported by `python -m vla.export` and reproduces the PyTorch
// forward pass: CoordConv image stem, language-conditioned (FiLM) spatial-softmax
// keypoints, word and proprioception embeddings, a pre-norm Transformer encoder
// with masked multi-head attention, and the action head.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vla {

struct Config {
  int dim = 0, depth = 0, heads = 0, mlp_dim = 0, patch = 0, chunk = 0, action_dim = 0;
  int proprio_dim = 0, vocab = 0, max_tokens = 0, img_h = 0, img_w = 0, stem1 = 0, stem2 = 0;
  int n_keypoints = 0;

  int n_patches() const { return (img_h / patch) * (img_w / patch); }
  int seq_len() const { return 3 + max_tokens + n_patches(); }
  int output_size() const { return chunk * action_dim; }
};

struct LayerWeights {
  std::vector<float> ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b;
  std::vector<float> ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b;
};

class Policy {
 public:
  // Throws std::runtime_error if the file is missing, truncated or malformed.
  explicit Policy(const std::string& weights_path);

  const Config& config() const { return cfg_; }

  // Keypoints (u, v) in [-1, 1] from the most recent predict(): named block, named pad, gripper.
  const std::vector<float>& last_keypoints() const { return keypoints_; }

  // image:   img_h * img_w * 3 bytes, row-major HWC RGB
  // tokens:  max_tokens word ids (0 = padding, ignored by attention)
  // proprio: proprio_dim floats
  // returns: chunk * action_dim normalised actions in [-1, 1] (row-major, chunk first)
  std::vector<float> predict(const uint8_t* image, const int32_t* tokens, const float* proprio);

 private:
  void linear(const float* x, int rows, int in, const std::vector<float>& w,
              const std::vector<float>& b, int out, float* y) const;
  void layer_norm(const float* x, int rows, const std::vector<float>& w,
                  const std::vector<float>& b, float* y) const;
  void attention(const float* qkv, const std::vector<uint8_t>& key_mask, float* out, int n_queries);
  // 2-D convolution via im2col; input and output are channel-major (C, H, W).
  void conv2d(const float* in, int c_in, int h, int w, const std::vector<float>& weight,
              const std::vector<float>& bias, int c_out, int k, int stride, int pad, float* out,
              int* h_out, int* w_out);

  Config cfg_;
  std::vector<float> conv1_w_, conv1_b_, conv2_w_, conv2_b_, conv3_w_, conv3_b_;
  float kp_scale_ = 1.f;
  std::vector<float> film_w_, film_b_, kp_conv_w_, kp_conv_b_, kp_embed_w_, kp_embed_b_, word_emb_, proprio_w_, proprio_b_, action_query_, pos_;
  std::vector<LayerWeights> layers_;
  std::vector<float> ln_out_w_, ln_out_b_, head_w_, head_b_;

  // Scratch buffers, allocated once so predict() does not allocate per token.
  std::vector<float> x_, norm_, qkv_, att_, tmp_, hidden_;
  std::vector<float> input_, feat1_, feat2_, feat3_, cols_, filmed_, heat_, pix_;
  std::vector<float> q_, kt_, v_, scores_, head_out_;   // per-head attention buffers
  std::vector<uint8_t> key_mask_;
  std::vector<float> keypoints_;
};

}  // namespace vla
