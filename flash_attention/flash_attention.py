import torch
import triton
import triton.language as tl
from flash_attention.forward_kernel import _attn_fwd


class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale, mask = None):
        # Asserts
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        O = torch.empty_like(Q)
        grid = lambda args: (
            BATCH_SIZE, # dim 0: batch index
            NUM_HEADS, # dim 1: head index
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # dim 2: query blocks
        )

        # Logsumexpr
        L = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )
        if mask:
            assert mask.dim() == 4, "Mask must be 4-dimensional (batch, head, seq1, seq2)"
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            L=L,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM,
            IS_MASK = False if mask is None else True,
            MASK=torch.zeros(1,1) if mask is None else mask,
            stride_MASK_batch=mask.stride(0) if mask is not None else 0,
            stride_MASK_head=mask.stride(1) if mask is not None else 0,
            stride_MASK_seq1=mask.stride(2) if mask is not None else 0,
            stride_MASK_seq2=mask.stride(3) if mask is not None else 0,
        )

        ctx.save_for_backward(Q, K, V, O, L, mask)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        return O

    def backward():
        pass

def test_flash_attention_forward():
    BATCH_SIZE = 2
    NUM_HEADS = 4
    SEQ_LEN = 32
    HEAD_DIM = 16
    dtype = torch.float16
    causal = False

    # Create random inputs
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)

    # Compute reference output using PyTorch
    Q_ = Q.to(torch.float32)
    K_ = K.to(torch.float32)
    V_ = V.to(torch.float32)
    attn_weights = torch.matmul(Q_, K_.transpose(-2, -1)) / softmax_scale  # (B, H, L, L)
    if causal:
        mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=Q.device))
        attn_weights = attn_weights.masked_fill(mask[None,None,:,:]==0, float('-inf'))
    attn_probs = torch.softmax(attn_weights, dim=-1)
    ref_output = torch.matmul(attn_probs, V_).to(dtype)
    try:
        mask = None
        if causal:
            mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=Q.device))
        tri_output = TritonAttention.apply(Q, K, V, softmax_scale, mask)
        tri_output = tri_output.to(dtype)
    except Exception as e:
        print("Error running TritonAttention:", e)
        return False

    # Compare the outputs
    max_diff = (ref_output - tri_output).abs().max().item()
    print(f"Max difference between reference and flash attention: {max_diff:.6e}")
    assert max_diff < 2e-2, f"Forward outputs do not match (max diff {max_diff})"
    print("Test passed: forward outputs match within acceptable tolerance.")




def compare(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal: bool, dtype = torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.randn_like(Q)

    # softmax(QK^T / scale) * V
    causal_mask = torch.tril((SEQ_LEN, SEQ_LEN), device = 'cuda')
    P = Q @ K.transpose(2, 3) / softmax_scale # (B, H, L, L)
    if causal:
        P[:, :, causal_mask == 0] = float('-inf')
    P = torch.softmax(P, dim = 3).half()
    O = P @ V
    # Backwards
    O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None


    # My implementation (soon)
    # tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    # tri_out.backward(dO)
    # tri_dV, V.grad = V.grad.clone(), None
    # tri_dK, K.grad = K.grad.clone(), None
    # tri_dQ, Q.grad = Q.grad.clone(), None
     
    # Compare
    # rtol = 0.0
    # atol = 1e-2
    # assert torch.allclose(ref_O, tri_out, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
