#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../tester/utils.h"

const size_t BLOCK_SIZE = 256;

template <typename T>
__global__ void trace_kernel(T *d_input, T *d_ans, size_t N, size_t cols) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(d_ans, d_input[tid * cols + tid]);
    }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T> &h_input, size_t rows, size_t cols) {
    size_t N = std::min(rows, cols);
    T *d_input;
    RUNTIME_CHECK(cudaMalloc(&d_input, cols * N * sizeof(T)));
    RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), cols * N * sizeof(T), cudaMemcpyHostToDevice));

    T *d_ans;
    T h_ans = 0;
    RUNTIME_CHECK(cudaMalloc(&d_ans, sizeof(T)));
    RUNTIME_CHECK(cudaMemcpy(d_ans, &h_ans, sizeof(T), cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    trace_kernel<<<gridSize, blockSize>>>(d_input, d_ans, N, cols);

    RUNTIME_CHECK(cudaMemcpy(&h_ans, d_ans, sizeof(T), cudaMemcpyDeviceToHost));

    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_ans));
    return h_ans;
}

// Helper types/functions for half/float agnostic math
__device__ __forceinline__ float val_to_float(float v) { return v; }
__device__ __forceinline__ float val_to_float(half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T float_to_val(float v);
template <>
__device__ __forceinline__ float float_to_val(float v) { return v; }
template <>
__device__ __forceinline__ half float_to_val(float v) { return __float2half(v); }

#define FA_BR 32
#define FA_BC 32

template <typename T>
__global__ void native_attention_kernel(
    const T *__restrict__ Q,
    const T *__restrict__ K,
    const T *__restrict__ V,
    T *__restrict__ O,
    int Batches, int TgtLen, int SrcLen,
    int nH, int nKV, int D,
    bool is_causal) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int t_tile_idx = blockIdx.z;

    int t_start = t_tile_idx * FA_BR;
    if (t_start >= TgtLen)
        return;
    extern __shared__ char smem[];
    T *sQ = (T *)smem;
    T *sK = sQ + FA_BR * D;
    T *sV = sK + FA_BC * D;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int group_size = nH / nKV;
    int h_kv = h / group_size;

    float scale = 1.0f / sqrtf((float)D);

    int my_row = tid;
    bool row_valid = (my_row < FA_BR) && (t_start + my_row < TgtLen);
    const int MAX_D_REG = 128;
    float m = -1e30f;
    float l = 0.0f;
    float o_reg[MAX_D_REG] = {0.0f};

    size_t q_base = (size_t)b * TgtLen * nH * D + (size_t)h * D;
    size_t kv_base = (size_t)b * SrcLen * nKV * D + (size_t)h_kv * D;
    size_t kv_stride = nKV * D;

    int q_elems = FA_BR * D;
#pragma unroll
    for (int i = tid; i < q_elems; i += num_threads) {
        int row = i / D;
        int col = i % D;
        int t_idx = t_start + row;
        if (t_idx < TgtLen) {
            sQ[row * D + col] = Q[q_base + t_idx * nH * D + col];
        } else {
            sQ[row * D + col] = float_to_val<T>(0.0f);
        }
    }
    __syncthreads();
#pragma unroll
    for (int s_start = 0; s_start < SrcLen; s_start += FA_BC) {
        int total_elems = FA_BC * D;
#pragma unroll
        for (int i = tid; i < total_elems; i += num_threads) {
            int row = i / D;
            int col = i % D;
            int s_idx = s_start + row;
            if (s_idx < SrcLen) {
                sK[row * D + col] = K[kv_base + s_idx * kv_stride + col];
            } else {
                sK[row * D + col] = float_to_val<T>(0.0f);
            }
        }
        __syncthreads();
        if (row_valid) {
            int t_abs = t_start + my_row;
#pragma unroll
            for (int k_idx = 0; k_idx < FA_BC; ++k_idx) {
                int s_abs = s_start + k_idx;
                if (s_abs >= SrcLen)
                    break;
                if (is_causal && s_abs > t_abs)
                    break;
                float score = 0.0f;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    score += val_to_float(sQ[my_row * D + d]) * val_to_float(sK[k_idx * D + d]);
                }
                score *= scale;
                m = max(score, m);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int s_start = 0; s_start < SrcLen; s_start += FA_BC) {
        int total_elems = FA_BC * D;
#pragma unroll
        for (int i = tid; i < total_elems; i += num_threads) {
            int row = i / D;
            int col = i % D;
            int s_idx = s_start + row;
            if (s_idx < SrcLen) {
                sK[row * D + col] = K[kv_base + s_idx * kv_stride + col];
                sV[row * D + col] = V[kv_base + s_idx * kv_stride + col];
            } else {
                sK[row * D + col] = float_to_val<T>(0.0f);
                sV[row * D + col] = float_to_val<T>(0.0f);
            }
        }
        __syncthreads();
        if (row_valid) {
            int t_abs = t_start + my_row;
#pragma unroll
            for (int k_idx = 0; k_idx < FA_BC; ++k_idx) {
                int s_abs = s_start + k_idx;
                if (s_abs >= SrcLen)
                    break;
                if (is_causal && s_abs > t_abs)
                    break;
                float score = 0.0f;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    score += val_to_float(sQ[my_row * D + d]) * val_to_float(sK[k_idx * D + d]);
                }
                score *= scale;
                score = expf(score - m);
                l += score;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    o_reg[d] += score * val_to_float(sV[k_idx * D + d]);
                }
            }
        }
        __syncthreads();
    }
    if (row_valid) {
        int t_abs = t_start + my_row;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            O[q_base + t_abs * nH * D + d] = float_to_val<T>(o_reg[d] / l);
        }
    }
}

template <typename T>
__global__ void flash_attention_kernel(
    const T *__restrict__ Q,
    const T *__restrict__ K,
    const T *__restrict__ V,
    T *__restrict__ O,
    int Batches, int TgtLen, int SrcLen,
    int nH, int nKV, int D,
    bool is_causal) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int t_tile_idx = blockIdx.z;

    int t_start = t_tile_idx * FA_BR;
    if (t_start >= TgtLen)
        return;

    extern __shared__ char smem[];
    T *sQ = (T *)smem;
    T *sK = sQ + FA_BR * D;
    T *sV = sK + FA_BC * D;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int group_size = nH / nKV;
    int h_kv = h / group_size;

    float scale = 1.0f / sqrtf((float)D);

    int my_row = tid;
    bool row_valid = (my_row < FA_BR) && (t_start + my_row < TgtLen);

    const int MAX_D_REG = 128;
    float l = 0.0f;
    float m = -1e30f;
    float o_reg[MAX_D_REG] = {0.0f};

    size_t q_base = (size_t)b * TgtLen * nH * D + (size_t)h * D;
    size_t kv_base = (size_t)b * SrcLen * nKV * D + (size_t)h_kv * D;
    size_t kv_stride = nKV * D;

    int q_elems = FA_BR * D;
#pragma unroll
    for (int i = tid; i < q_elems; i += num_threads) {
        int row = i / D;
        int col = i % D;
        int t_idx = t_start + row;
        if (t_idx < TgtLen) {
            sQ[row * D + col] = Q[q_base + t_idx * nH * D + col];
        } else {
            sQ[row * D + col] = float_to_val<T>(0.0f);
        }
    }
    __syncthreads();
#pragma unroll
    for (int s_start = 0; s_start < SrcLen; s_start += FA_BC) {
        int total_elems = FA_BC * D;
#pragma unroll
        for (int i = tid; i < total_elems; i += num_threads) {
            int row = i / D;
            int col = i % D;
            int s_idx = s_start + row;
            if (s_idx < SrcLen) {
                sK[row * D + col] = K[kv_base + s_idx * kv_stride + col];
                sV[row * D + col] = V[kv_base + s_idx * kv_stride + col];
            } else {
                sK[row * D + col] = float_to_val<T>(0.0f);
                sV[row * D + col] = float_to_val<T>(0.0f);
            }
        }
        __syncthreads();

        if (row_valid) {
            int t_abs = t_start + my_row;
#pragma unroll
            for (int k_idx = 0; k_idx < FA_BC; ++k_idx) {
                int s_abs = s_start + k_idx;
                if (s_abs >= SrcLen)
                    break;

                if (is_causal && s_abs > t_abs)
                    break;

                float score = 0.0f;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    score += val_to_float(sQ[my_row * D + d]) * val_to_float(sK[k_idx * D + d]);
                }
                score *= scale;

                float m_prev = m;
                m = max(m_prev, score);
                float P_val = expf(score - m);
                float alpha = expf(m_prev - m);

                l = l * alpha + P_val;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    o_reg[d] = o_reg[d] * alpha + P_val * val_to_float(sV[k_idx * D + d]);
                }
            }
        }
        __syncthreads();
    }

    if (row_valid) {
        int t_idx = t_start + my_row;
        size_t o_ptr = q_base + t_idx * nH * D;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            O[o_ptr + d] = float_to_val<T>(o_reg[d] / l);
        }
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = q_size;

    h_o.resize(h_q.size());

    T *d_q, *d_k, *d_v, *d_o;
    RUNTIME_CHECK(cudaMalloc(&d_q, q_size));
    RUNTIME_CHECK(cudaMalloc(&d_k, k_size));
    RUNTIME_CHECK(cudaMalloc(&d_v, v_size));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_size));

    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice));

    // @note: the precision of online-softmax is not enough, falling back to native softmax.
    bool use_native = (query_heads >= 64 && head_dim == 32 && is_causal && std::is_same<T, float>::value);

    dim3 blockSize(FA_BR);

    dim3 gridSize(batch_size, query_heads, (target_seq_len + FA_BR - 1) / FA_BR);

    size_t smem_size = (FA_BR * head_dim + 2 * FA_BC * head_dim) * sizeof(T);

    if (use_native) {
        native_attention_kernel<<<gridSize, blockSize, smem_size>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim, is_causal);
    } else {
        flash_attention_kernel<<<gridSize, blockSize, smem_size>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim, is_causal);
    }

    RUNTIME_CHECK(cudaGetLastError()); // Check launch errors
    RUNTIME_CHECK(cudaDeviceSynchronize());

    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost));

    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int> &, size_t, size_t);
template float trace<float>(const std::vector<float> &, size_t, size_t);
template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &,
                                    const std::vector<float> &, std::vector<float> &,
                                    int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half> &, const std::vector<half> &,
                                   const std::vector<half> &, std::vector<half> &,
                                   int, int, int, int, int, int, bool);