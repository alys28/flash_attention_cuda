import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    o,
    l_i,
    m_i,
    q,
    k_block_ptr,
    v_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    for i in range(0, SEQ_LEN, BLOCK_SIZE_KV):
        # Load one block at a time. Cannot load all of k (SEQ_LEN, HEAD_DIM) (too much memory for Shared memory) 
        # -> Instead load (BLOCK_SIZE_KV, HEAD_DIM) at a time, and use the magic of online softmax!!
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        s_i = tl.dot(q, k)
        s_i /= softmax_scale
        row_max = tl.max(s_i, axis = 1)
        m_i_1 = tl.maximum(m_i, row_max)
        p_i = tl.exp(s_i - m_i_1)
        l_i_1 = tl.sum(p_i, axis = 1) + l_i * tl.exp(m_i - m_i_1) # m_i_1 will be broadcasted from shape (BLOCK_SIZE_Q, ) to (BLOCK_SIZE_Q, BLOCK_SIZE_KV) to add with s_i
        # Create diagonal matrix
        product = tl.dot(p_i, v)
        scale = tl.exp(m_i - m_i_1)
        o = o * scale[:, None] + product # O has shape (BLOCK_SIZE_Q, HEAD_DIM)
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_SIZE_KV))
        v_block_ptr = tl.advance(v_block_ptr, (0, BLOCK_SIZE_KV))
        l_i = l_i_1
        m_i = m_i_1
    return o, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    L,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
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

    v_base_offset = idx_batch * stride_V_batch + idx_head * stride_V_head 
    v_block_ptr = tl.make_block_ptr(
        base = V + v_base_offset,
        shape = (SEQ_LEN, HEAD_DIM), # Transpose
        strides = (stride_V_seq, stride_V_dim), 
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1, 0) # We want the first dimension to have contiguous elements in memory 
    )

    o_block_ptr = tl.make_block_ptr(
        base = O,
        shape = (SEQ_LEN, HEAD_DIM), # Transpose
        strides = (stride_O_seq, stride_O_dim), 
        offsets = (idx_q_block * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0) # We want the first dimension to have contiguous elements in memory 
    )
    o = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    l_i = tl.zeros((BLOCK_SIZE_Q, ), dtype=tl.float32)
    m_i = tl.fill((BLOCK_SIZE_Q, ), float("-inf"), dtype=tl.float32)

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            o,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            idx_q_block,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            0, # offs_q ,
            0, # offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            o,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            idx_q_block,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            0, # offs_q,
            0, # offs_kv,
            SEQ_LEN,
        )
    # epilogue
    L_i = m_i + tl.math.log(l_i)  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None] # Divid each entry by the corresponding row element in l_i
    l_ptrs = L + (idx_batch * NUM_HEADS + idx_head) * SEQ_LEN + BLOCK_SIZE_Q * idx_q_block + tl.arange(0, BLOCK_SIZE_Q) # One scalar L_i per query row
    tl.store(l_ptrs, L_i)
    tl.store(o_block_ptr, O_block.to(O.type.element_ty))
