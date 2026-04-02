# Flash Attention (Triton)

This repository contains an implementation of Flash Attention using Triton kernels.
It is currently focused on learning and experimentation, not production use.

## What is implemented

- Forward + backward pass kernel in Triton with online softmax accumulation.
- Optional attention mask support (`bool` or `0/1` inputs converted to additive `0/-inf`).
- A forward correctness check against a PyTorch reference implementation.

## Repository layout

- `flash_attention/flash_attention.py`: Autograd wrapper (`TritonAttention`) and a forward test entrypoint.
- `flash_attention/forward_kernel.py`: Triton forward kernels.
- `flash_attention/backward_kernel.py`: Backward preprocessing kernel
- `CUDA_kernel_examples/`: Standalone CUDA learning examples.
- `Triton_examples/`: Standalone Triton learning notebook(s).

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- PyTorch built with CUDA
- Triton

Install dependencies in your environment:

```bash
pip install torch triton
```

## Quickstart

Minimal usage of the forward pass:

```python
import torch
from flash_attention.flash_attention import TritonAttention

B, H, L, D = 2, 4, 128, 64
Q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
K = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)
V = torch.randn(B, H, L, D, device="cuda", dtype=torch.float32, requires_grad=True)

softmax_scale = D ** -0.5
mask = torch.tril(torch.ones(B, H, L, L, device="cuda", dtype=torch.bool))

O = TritonAttention.apply(Q, K, V, softmax_scale, mask)  # shape: (B, H, L, D)
```

## Run the included forward test

From the repository root:

```bash
python3 -m flash_attention.flash_attention
```

The script compares Triton output to a PyTorch reference and asserts that the max difference is within tolerance.

## Notes and limitations

- Inputs are expected in `(batch, heads, seq_len, head_dim)` layout.
- `Q`, `K`, and `V` must use the same `head_dim`.
- This codebase is actively evolving; API and kernel details may change.
- Future add-on: RoPE embeddings
