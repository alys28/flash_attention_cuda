import torch
import triton
import triton.language as tl



@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    idx_batch = tl.program_id(0)
    idx_head = tl.program_id(1)
    idx_q_block = tl.program_id(2)
    # We have to go the amount of strides to reach the corresponding block within the head
    base_offset = idx_batch * stride_Q_batch + idx_head * stride_Q_head # Get to the right batch, find the right head, you will have the sub matrix of shape (SEQ_LEN, HEAD_DIM)
    q_block_ptr = tl.make_block_ptr(
        base = Q + base_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_Q_seq, stride_Q_dim), 
        offsets = (idx_q_block * BLOCK_SIZE_Q, 0), # Top left of our matrix basically, not a range, but rather coordinates
        shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0) # We want the first dimension to have contiguous elements in memory
    )
    # # From scratch implementation of q_block_ptr:
    # q_base = Q + base_offset
    # offset_row = idx_q_block * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offset_col = tl.arange(0, HEAD_DIM)
    # q_ptrs = q_base + offset_row[:, None] * stride_Q_seq + offset_col[None, :] * stride_Q_dim
    # mask = offset_row < SEQ_LEN

    q = tl.load(q_block_ptr)

    
    


class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        # Asserts
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        O = torch.empty_like(Q)
        stage = 3 if causal else 1
        grid = lambda args: (
            BATCH_SIZE, # dim 0: batch index
            NUM_HEADS, # dim 1: head index
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # dim 2: query blocks
        )

        # Logsumexpr
        L = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

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
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O

    def backward():
        pass




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
