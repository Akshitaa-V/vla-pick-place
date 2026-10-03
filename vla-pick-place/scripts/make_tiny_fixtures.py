"""Write a small randomly initialised model and its PyTorch outputs for the C++ tests."""
import torch

from vla.export import export_fixtures, export_weights
from vla.model import PolicyConfig, VLAPolicy

torch.manual_seed(0)
model = VLAPolicy(PolicyConfig(dim=32, depth=2, heads=2, mlp_dim=64)).eval()
with torch.no_grad():                      # make every tensor non-trivial
    model.pos.normal_(0, 0.5)
    model.action_query.normal_(0, 0.5)
    model.film.weight.normal_(0, 0.2)                 # FiLM starts at identity; perturb it
    model.film.bias.normal_(0, 0.2)
export_weights(model, "cpp/tests/tiny_policy.bin")
export_fixtures(model, "cpp/tests/tiny_fixtures.bin", n=8)
print("wrote cpp/tests/tiny_policy.bin and cpp/tests/tiny_fixtures.bin")
