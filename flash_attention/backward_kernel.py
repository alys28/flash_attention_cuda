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
    dOi = tl.load(O + batch_head_idx * HEAD_DIM * SEQ_LEN + offset_q[:, None] + offset_head[None, :])
    D_i = tl.sum(dOi * O_i, axis=1)
    tl.store(D + batch_head_idx * SEQ_LEN + offset_q, D_i)