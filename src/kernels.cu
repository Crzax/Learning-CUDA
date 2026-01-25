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
    // TODO: Implement the flash attention function
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