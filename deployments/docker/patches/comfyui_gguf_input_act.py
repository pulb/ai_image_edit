"""Make ComfyUI-GGUF's GGMLOps.Linear accept the fused-activation keyword
arguments that ComfyUI >= 0.39 passes to every Linear layer (input_act,
act_weight, act_eps, residual, residual_scale). Upstream ops.py has CRLF line
endings, so the file is handled without newline translation.

Fails (non-zero exit) if the expected original code is not found, so a changed
upstream is noticed at build time instead of at the first generation.
"""
import sys

path = sys.argv[1]
with open(path, newline="") as f:
    src = f.read()

old = (
    "        def forward_ggml_cast_weights(self, input):\r\n"
    "            weight, bias = self.cast_bias_weight(input)\r\n"
    "            return torch.nn.functional.linear(input, weight, bias)\r\n"
)
new = (
    "        def forward_ggml_cast_weights(self, input, input_act=None, act_weight=None,\r\n"
    "                                      act_eps=0.0, residual=None, residual_scale=None):\r\n"
    "            input = comfy.ops._eager_input_act(input, input_act, act_weight, act_eps)\r\n"
    "            weight, bias = self.cast_bias_weight(input)\r\n"
    "            out = torch.nn.functional.linear(input, weight, bias)\r\n"
    "            return comfy.ops._linear_residual(out, residual, residual_scale)\r\n"
)
if src.count(old) != 1:
    sys.exit("comfyui_gguf_input_act: expected GGMLOps.Linear code not found in " + path)
with open(path, "w", newline="") as f:
    f.write(src.replace(old, new))
print("patched", path)
