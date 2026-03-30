import torch

import triton
import triton.language as tl



@triton.jit
def _attn_bwd_preprocess(
    O, # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    dO, # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    D, # (BATCH_SIZE, NUM_HEADS, SEQ_LEN)
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_idx_q = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    offset_q = BLOCK_SIZE_Q * block_idx_q + tl.arange(0, BLOCK_SIZE_Q)
    offset_head = tl.arange(0, HEAD_DIM)
    O_i = tl.load(O + batch_head_idx * HEAD_DIM * SEQ_LEN + offset_q[:, None] + offset_head[None, :])
    dOi = tl.load(dO + batch_head_idx * HEAD_DIM * SEQ_LEN + offset_q[:, None] + offset_head[None, :])
    D_i = tl.sum(dOi * O_i, axis=1)
    tl.store(D + batch_head_idx * SEQ_LEN + offset_q, D_i)



@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    L,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_MASK: tl.constexpr,
    MASK,
    stride_MASK_batch,
    stride_MASK_head,
    stride_MASK_seq1,
    stride_MASK_seq2,
):
    batch_head_idx = tl.program_id(1)
    batch_idx = batch_head_idx // NUM_HEADS
    head_idx = batch_head_idx % NUM_HEADS
    block_idx_kv = tl.program_id(0)
    dv_j = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    dk_j = tl.zeros((BLOCK_KV, HEAD_DIM), dtype=tl.float32)
    k_base = block_idx_kv * BLOCK_KV * stride_seq + (batch_head_idx // NUM_HEADS) * stride_batch + (batch_head_idx % NUM_HEADS) * stride_head
    v_base = block_idx_kv * BLOCK_KV * stride_seq + (batch_head_idx // NUM_HEADS) * stride_batch + (batch_head_idx % NUM_HEADS) * stride_head
    k_j = tl.load(K + k_base + tl.arange(0, BLOCK_KV)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim) # (BLOCK_KV, HEAD_DIM)
    v_j = tl.load(V + v_base + tl.arange(0, BLOCK_KV)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim) # (BLOCK_KV, HEAD_DIM)
    kv_mask = tl.arange(0, BLOCK_KV) + block_idx_kv * BLOCK_KV < SEQ_LEN
    for q_block_start in range(0, SEQ_LEN, BLOCK_Q):
        # Compute dK and dV for the current block of queries
        q_positions = q_block_start + tl.arange(0, BLOCK_Q) # (BLOCK_Q,)
        q_mask = q_positions < SEQ_LEN
        q_base = batch_head_idx // NUM_HEADS * stride_batch + (batch_head_idx % NUM_HEADS) * stride_head + q_block_start * stride_seq
        q_i = tl.load(Q + q_base + tl.arange(0, BLOCK_Q)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim) # (BLOCK_Q, HEAD_DIM)
        dO_i = tl.load(dO + q_base + tl.arange(0, BLOCK_Q)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim) # (BLOCK_Q, HEAD_DIM)
        L_i = tl.load(L + batch_idx * NUM_HEADS * SEQ_LEN + head_idx * SEQ_LEN + q_block_start + tl.arange(0, BLOCK_Q)[:, None]) # (BLOCK_Q, 1) 
        p_ij = tl.exp(q_i @ k_j.transpose(0, 1) * softmax_scale)  / L_i # (BLOCK_Q, BLOCK_KV)
        p_ij = tl.where(q_mask[:, None], p_ij, 0)
        p_ij = tl.where(kv_mask[None, :], p_ij, 0)
        if IS_MASK:
            mask_ij = tl.load(MASK + batch_idx * stride_MASK_batch + head_idx * stride_MASK_head + (q_block_start + tl.arange(0, BLOCK_Q)[:, None]) * stride_MASK_seq1 + (block_idx_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)[None, :]) * stride_MASK_seq2)
            p_ij = p_ij * mask_ij
        dv_j += p_ij.transpose(0, 1) @ dO_i # (BLOCK_KV, HEAD_DIM)
        D_base = batch_idx * NUM_HEADS * SEQ_LEN + head_idx * SEQ_LEN + q_block_start + tl.arange(0, BLOCK_Q)[:,None] # (BLOCK_Q, 1)
        D_i = tl.load(D + D_base) # (BLOCK_Q, 1)
        w_ij = p_ij * (dO_i @ v_j.transpose(0, 1) - D_i) # (BLOCK_Q, BLOCK_KV) -> element wise product
        dk_j += w_ij.transpose(0, 1) @ q_i # (BLOCK_KV, HEAD_DIM)
    
    # Write dK and dV for the current block of keys/values
    tl.store(dK + k_base + tl.arange(0, BLOCK_KV)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim, dk_j, mask=kv_mask[:, None])
    tl.store(dV + v_base + tl.arange(0, BLOCK_KV)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim, dv_j, mask=kv_mask[:, None])



@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    L,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_MASK: tl.constexpr,
    MASK,
    stride_MASK_batch,
    stride_MASK_head,
    stride_MASK_seq1,
    stride_MASK_seq2,
):
    batch_head_idx = tl.program_id(1)
    batch_idx = batch_head_idx // NUM_HEADS
    head_idx = batch_head_idx % NUM_HEADS
    block_idx_q = tl.program_id(0)
    q_base = batch_idx * stride_batch + head_idx * stride_head + block_idx_q * BLOCK_Q * stride_seq
    q_i = tl.load(Q + q_base + tl.arange(0, BLOCK_Q)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim) # (BLOCK_Q, HEAD_DIM)
    D_i = tl.load(D + batch_idx * NUM_HEADS * SEQ_LEN + head_idx * SEQ_LEN + block_idx_q * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]) # (BLOCK_Q, 1)
    L_i = tl.load(L + batch_idx * NUM_HEADS * SEQ_LEN + head_idx * SEQ_LEN + block_idx_q * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]) # (BLOCK_Q, 1)
    dO_i = tl.load(dO + q_base + tl.arange(0, BLOCK_Q)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim) # (BLOCK_Q, HEAD_DIM)
    dq_i = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)
    q_mask = tl.arange(0, BLOCK_Q) + block_idx_q * BLOCK_Q < SEQ_LEN
    for block_start_kv in range(0, SEQ_LEN, BLOCK_KV):
        kv_base = batch_idx * stride_batch + head_idx * stride_head + block_start_kv * stride_seq
        k_j = tl.load(K + kv_base + tl.arange(0, BLOCK_KV)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim) # (BLOCK_KV, HEAD_DIM)
        v_j = tl.load(V + kv_base + tl.arange(0, BLOCK_KV)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim) # (BLOCK_KV, HEAD_DIM)
        kv_mask = tl.arange(0, BLOCK_KV) + block_start_kv * BLOCK_KV < SEQ_LEN
        p_ij = tl.exp(q_i @ k_j.transpose(0, 1) * softmax_scale) / L_i # (BLOCK_Q, BLOCK_KV)
        p_ij = tl.where(q_mask[:, None], p_ij, 0)
        p_ij = tl.where(kv_mask[None, :], p_ij, 0)
        if IS_MASK:
            mask_ij = tl.load(MASK + batch_idx * stride_MASK_batch + head_idx * stride_MASK_head + (block_idx_q * BLOCK_Q + tl.arange(0, BLOCK_Q)[:, None]) * stride_MASK_seq1 + (block_start_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)[None, :]) * stride_MASK_seq2)
            p_ij = p_ij * mask_ij
        w_ij = p_ij * (dO_i @ v_j.transpose(0, 1) - D_i) # (BLOCK_Q, HEAD_DIM)
        dq_i += w_ij @ k_j # (BLOCK_Q, HEAD_DIM)
    tl.store(dQ + q_base + tl.arange(0, BLOCK_Q)[:, None] * stride_seq + tl.arange(0, HEAD_DIM)[None, :] * stride_dim, dq_i, mask=q_mask[:, None])