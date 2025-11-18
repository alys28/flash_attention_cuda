import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    pass

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
    q_base_offset = idx_batch * stride_Q_batch + idx_head * stride_Q_head # Get to the right batch, find the right head, you will have the sub matrix of shape (SEQ_LEN, HEAD_DIM)
    q_block_ptr = tl.make_block_ptr(
        base = Q + q_base_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_Q_seq, stride_Q_dim), 
        offsets = (idx_q_block * BLOCK_SIZE_Q, 0), # Top left of our matrix basically, not a range, but rather coordinates
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0) # We want the first dimension to have contiguous elements in memory
    )
    # # From scratch implementation of q_block_ptr:
    # q_base = Q + base_offset
    # offset_row = idx_q_block * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # offset_col = tl.arange(0, HEAD_DIM)
    # q_ptrs = q_base + offset_row[:, None] * stride_Q_seq + offset_col[None, :] * stride_Q_dim
    # mask = offset_row < SEQ_LEN

    q = tl.load(q_block_ptr)

    # Load K
    k_base_offset = idx_batch * stride_K_batch + idx_head * stride_K_head 
    k_block_ptr = tl.make_block_ptr(
        base = K + k_base_offset,
        shape = (HEAD_DIM, SEQ_LEN), # Transpose
        strides = (stride_K_dim, stride_K_seq), 
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order = (0, 1) # We want the first dimension to have contiguous elements in memory 
    )
    k = tl.load(k_block_ptr)


    v_base_offset = idx_batch * stride_V_batch + idx_head * stride_V_head 
    v_block_ptr = tl.make_block_ptr(
        base = V + v_base_offset,
        shape = (SEQ_LEN, HEAD_DIM), # Transpose
        strides = (stride_V_dim, stride_V_seq), 
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1, 0) # We want the first dimension to have contiguous elements in memory 
    )
    v = tl.load(v_block_ptr)

    o_block_ptr = tl.make_block_ptr(
        base = O,
        shape = (SEQ_LEN, HEAD_DIM), # Transpose
        strides = (stride_O_dim, stride_O_seq), 
        offsets = (idx_q_block * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0) # We want the first dimension to have contiguous elements in memory 
    )
    o = tl.load(v_block_ptr)

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            q,
            k,
            v,
            idx_q_block,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            q,
            k,
            v,
            idx_q_block,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    # epilogue
    m_i += tl.math.log(
        l_i
    )  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))
