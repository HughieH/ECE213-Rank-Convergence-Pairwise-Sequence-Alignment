/*
    Tiled NW with Rank Convergence (3-kernel approach)

    RC mapping:

        forwardPass -> Parallal Forward Pass kernel: numPairs * P blocks, block = pair * P + c owns chunk c
        fixupPhase -> Fix-up kernel: numPairs * (P-1) blocks, iterates <= P-1 times
        tracebackPhase -> Traceback kernel: numPairs blocks (Similar to baseline implementation)

    TILE = 128 keeps smem at around 3.9KB per block
    nz (speculative start non-zero vector) = all +1 as mentioned in the paper
*/

#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <tbb/parallel_for.h>

#define TILE 128 // Dimensions of each tile, tiles make up the entire M x N DP matrix
#define P 4 // Number of chunks, aka stages in our RC algorithm 

// Protein scoring: constant memory tables
__device__ __constant__ int8_t d_aa_to_idx[256];
__device__ __constant__ int8_t d_blosum62[20*20];

static void initProteinTablesOnce() {
    static bool inited = false;
    if (inited) return;
    inited = true;

    int8_t aa_to_idx[256];
    for (int i = 0; i < 256; i++) aa_to_idx[i] = -1;
    const char* AA = "ARNDCQEGHILKMFPSTWYV";
    for (int i = 0; i < 20; i++)
        aa_to_idx[(unsigned char)AA[i]] = (int8_t)i;

    static const int8_t blosum62[20 * 20] = {
         4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,
        -1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,
        -2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3,
        -2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3,
         0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,
        -1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2,
        -1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2,
         0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,
        -2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3,
        -1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,
        -1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,
        -1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2,
        -1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,
        -2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,
        -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,
         1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2,
         0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,
        -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,
        -2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,
         0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4
    };

    cudaMemcpyToSymbol(d_aa_to_idx, aa_to_idx, sizeof(aa_to_idx));
    cudaMemcpyToSymbol(d_blosum62,  blosum62,  sizeof(blosum62));
}

__device__ __forceinline__ int32_t subScore(char r, char q, int32_t isProtein) {
    const int32_t MATCH    =  2;
    const int32_t MISMATCH = -1;
    if (!isProtein) return (r == q) ? MATCH : MISMATCH;
    int8_t ri = d_aa_to_idx[(unsigned char)r];
    int8_t qi = d_aa_to_idx[(unsigned char)q];
    return (ri >= 0 && qi >= 0) ? (int32_t)d_blosum62[ri * 20 + qi] : MISMATCH;
}

// ---------------------------------------------------------------------------
// Shared memory layout (TILE=128, 3,884 bytes):
//   shared_ref   [TILE]        char        128
//   shared_qry   [TILE]        char        128
//   pad          [8]           char          8
//   s_top        [TILE+1]      int32_t     516
//   s_left       [TILE+1]      int32_t     516
//   s_bot        [TILE+1]      int32_t     516
//   s_right      [TILE+1]      int32_t     516
//   wf_scores    [3*(TILE+1)]  int32_t    1548
//   smem_conv    [1]           int32_t       4  (fixup/traceback only)
//   TOTAL                                3,880
// ---------------------------------------------------------------------------
__device__ void computeTile(
    int tM, int tN,
    int32_t* __restrict__ s_top,
    int32_t* __restrict__ s_left,
    int32_t* __restrict__ s_bot,
    int32_t* __restrict__ s_right,
    int32_t* __restrict__ wf_scores,
    char*    __restrict__ shared_ref,
    char*    __restrict__ shared_qry,
    int32_t* __restrict__ tile_dp,
    int      DP_TILE_STRIDE,
    int32_t  isProtein,
    int tx, int bdx
) {
    const int32_t NEG_INF  = -1000000;
    const int32_t GAP      = -2;
    const int     WFS      = TILE + 1;
    const uint8_t DIR_DIAG = 1, DIR_UP = 2, DIR_LEFT = 3;

    for (int idx = tx; idx < 3*WFS; idx += bdx) wf_scores[idx] = NEG_INF;
    for (int idx = tx; idx <= tM;   idx += bdx) s_right[idx]   = NEG_INF;
    for (int idx = tx; idx <= tN;   idx += bdx) s_bot[idx]     = NEG_INF;

    if (tile_dp) {
        for (int i = tx; i <= tM; i += bdx) tile_dp[i * DP_TILE_STRIDE]     = s_left[i];
        for (int j = tx; j <= tN; j += bdx) tile_dp[0 * DP_TILE_STRIDE + j] = s_top[j];
    }
    __syncthreads();

    for (int k = 0; k <= tM + tN; ++k) {
        int curr_k   = (k     % 3) * WFS;
        int pre_k    = ((k+2) % 3) * WFS;
        int prepre_k = ((k+1) % 3) * WFS;

        for (int idx = tx; idx < WFS; idx += bdx)
            wf_scores[curr_k + idx] = NEG_INF;
        __syncthreads();

        int i_start = max(0,  k - tN);
        int i_end   = min(tM, k);

        for (int i = i_start + tx; i <= i_end; i += bdx) {
            int j = k - i;
            int32_t score; uint8_t dir = DIR_DIAG;

            if      (i == 0 && j == 0) { score = s_top[0]; }
            else if (i == 0)           { score = s_top[j]; }
            else if (j == 0)           { score = s_left[i]; }
            else {
                int32_t diag_base =
                    (i-1==0 && j-1==0) ? s_top[0]              :
                    (i-1==0)           ? s_top[j-1]            :
                    (j-1==0)           ? s_left[i-1]           :
                                        wf_scores[prepre_k+(i-1)];
                int32_t sd = diag_base + subScore(shared_ref[i-1], shared_qry[j-1], isProtein);
                int32_t su = ((i-1==0) ? s_top[j]  : wf_scores[pre_k+(i-1)]) + GAP;
                int32_t sl = ((j-1==0) ? s_left[i] : wf_scores[pre_k+i])     + GAP;
                score = sd; dir = DIR_DIAG;
                if (su > score) { score = su; dir = DIR_UP;   }
                if (sl > score) { score = sl; dir = DIR_LEFT; }
            }
            wf_scores[curr_k + i] = score;
            if (j == tN) s_right[i] = score;
            if (i == tM) s_bot[j]   = score;
            if (tile_dp) tile_dp[i * DP_TILE_STRIDE + j] = score;
        }
        __syncthreads();
    }
}

// ==========================================================================
// PHASE 1: Speculative forward pass
// Grid = numPairs * P, block (pair*P + c) owns chunk c of one pair.
// Chunk 0: uses true NW boundaries.
// Chunks 1..P-1: speculative — nz=1 for left column, top row = 0 (from memset).
// Per paper, nz must be all-non-zero so we use +1.
// ==========================================================================
__global__ void forwardPass(
    const int32_t* __restrict__ d_info,
    const int32_t* __restrict__ d_seqLen,
    const char*    __restrict__ d_seqs,
    int32_t*                    d_H_bnd,
    int32_t                     tile_rows_max,
    int32_t                     bnd_cols,
    int32_t*                    d_chunk_bnd,
    int32_t*                    d_delta_bnd,
    int32_t                     num_chunks
) {
    const int32_t GAP = -2;
    int tx  = threadIdx.x;
    int bdx = blockDim.x;

    int32_t numPairs  = d_info[0];
    int32_t maxSeqLen = d_info[1];
    int32_t isProtein = d_info[2];

    int pair = blockIdx.x / num_chunks;
    int c    = blockIdx.x % num_chunks;
    if (pair >= numPairs) return;

    int32_t M = d_seqLen[2 * pair];
    int32_t N = d_seqLen[2 * pair + 1];

    int32_t refStart = (int64_t)(pair * 2)     * maxSeqLen;
    int32_t qryStart = (int64_t)(pair * 2 + 1) * maxSeqLen;

    int32_t tile_rows  = (M + TILE - 1) / TILE;
    int32_t tile_cols  = (N + TILE - 1) / TILE;
    int     DP_TILE_STRIDE = TILE + 1;
    int     chunk_size = (tile_rows + num_chunks - 1) / num_chunks;

    int64_t hb_off = (int64_t)pair * (tile_rows_max + 1) * bnd_cols;
    int32_t* H_bnd = d_H_bnd + hb_off;

    int32_t* chunk_bnd = d_chunk_bnd + (int64_t)pair * num_chunks * bnd_cols;
    int32_t* delta_bnd = d_delta_bnd + (int64_t)pair * num_chunks * bnd_cols;

    int tr_start = c * chunk_size;
    int tr_end   = min(tr_start + chunk_size, tile_rows);
    if (tr_start >= tile_rows) return;

    extern __shared__ char smem[];
    char*    shared_ref = smem;
    char*    shared_qry = shared_ref + TILE;
    int32_t* s_top      = (int32_t*)(shared_qry + TILE + 8);
    int32_t* s_left     = s_top     + (TILE + 1);
    int32_t* s_bot      = s_left    + (TILE + 1);
    int32_t* s_right    = s_bot     + (TILE + 1);
    int32_t* wf_scores  = s_right   + (TILE + 1);

    // Chunk 0 writes the true NW base row into H_bnd[0..N]
    if (c == 0) {
        for (int j = tx; j <= N; j += bdx)
            H_bnd[j] = (int32_t)j * GAP;
        __syncthreads();
    }
    // Chunks 1..P-1: H_bnd[tr_start * bnd_cols + j] = 0 from cudaMemset — used as
    // speculative top boundary. We never overwrite it before reading it in the tr loop.

    for (int tr = tr_start; tr < tr_end; ++tr) {
        int row0 = tr * TILE;
        int tM   = min(TILE, M - row0);

        for (int tc = 0; tc < tile_cols; ++tc) {
            int col0 = tc * TILE;
            int tN   = min(TILE, N - col0);

            for (int i = tx; i < tM; i += bdx)
                shared_ref[i] = d_seqs[refStart + row0 + i];
            for (int j = tx; j < tN; j += bdx)
                shared_qry[j] = d_seqs[qryStart + col0 + j];

            // Top boundary from H_bnd (chunk 0: correct NW row; chunks 1..P-1: 0 from memset)
            for (int idx = tx; idx <= tN; idx += bdx)
                s_top[idx] = H_bnd[tr * bnd_cols + col0 + idx];

            if (tc == 0) {
                if (c == 0) {
                    // True NW left column: i * GAP
                    for (int idx = tx; idx <= tM; idx += bdx)
                        s_left[idx] = (int32_t)(row0 + idx) * GAP;
                } else {
                    // Speculative nz = +1 per §4.5 (all-non-zero, not -inf)
                    for (int idx = tx; idx <= tM; idx += bdx)
                        s_left[idx] = 1;
                }
            }
            __syncthreads();

            computeTile(tM, tN, s_top, s_left, s_bot, s_right,
                        wf_scores, shared_ref, shared_qry,
                        nullptr, DP_TILE_STRIDE, isProtein, tx, bdx);

            for (int idx = tx; idx <= tN; idx += bdx)
                H_bnd[(tr+1) * bnd_cols + col0 + idx] = s_bot[idx];
            for (int idx = tx; idx <= tM; idx += bdx)
                s_left[idx] = s_right[idx];
            __syncthreads();
        }
    }

    // Snapshot final output boundary
    for (int j = tx; j <= N; j += bdx)
        chunk_bnd[c * bnd_cols + j] = H_bnd[tr_end * bnd_cols + j];
    __syncthreads();

    // Compute and store baseline delta
    for (int j = tx; j <= N; j += bdx)
        delta_bnd[c * bnd_cols + j] = (j == 0)
            ? chunk_bnd[c * bnd_cols]
            : chunk_bnd[c * bnd_cols + j] - chunk_bnd[c * bnd_cols + j - 1];
}

// ==========================================================================
// PHASE 2: Fix-up iteration
// Grid = numPairs * (P-1). Block (pair*(P-1) + (c-1)) owns chunk c ∈ [1,P-1].
// Each kernel launch = one fix-up iteration (paper's do-while body).
// cudaDeviceSynchronize between launches = paper's barrier (line 25).
// delta_new is per (pair, c) to avoid race: d_delta_new[(pair*(P-1)+(c-1))*bnd_cols].
// ==========================================================================
__global__ void fixupPhase(
    const int32_t* __restrict__ d_info,
    const int32_t* __restrict__ d_seqLen,
    const char*    __restrict__ d_seqs,
    int32_t*                    d_H_bnd,
    int32_t                     tile_rows_max,
    int32_t                     bnd_cols,
    int32_t*                    d_chunk_bnd,
    int32_t*                    d_delta_bnd,
    int32_t*                    d_delta_new,   // [(P-1)*bnd_cols] per pair
    uint8_t*                    d_conv_flags,
    int32_t                     num_chunks
) {
    const int32_t GAP = -2;
    int tx  = threadIdx.x;
    int bdx = blockDim.x;

    int32_t numPairs  = d_info[0];
    int32_t maxSeqLen = d_info[1];
    int32_t isProtein = d_info[2];

    int pair = blockIdx.x / (num_chunks - 1);
    int c    = blockIdx.x % (num_chunks - 1) + 1;   // c ∈ [1, P-1]
    if (pair >= numPairs) return;

    uint8_t* conv_flags = d_conv_flags + pair * num_chunks;

    // Paper line 14: parallel.for p in (2..P) — always re-run unless both converged
    if (conv_flags[c] && conv_flags[c-1]) return;

    int32_t M = d_seqLen[2 * pair];
    int32_t N = d_seqLen[2 * pair + 1];

    int32_t refStart = (int64_t)(pair * 2)     * maxSeqLen;
    int32_t qryStart = (int64_t)(pair * 2 + 1) * maxSeqLen;

    int32_t tile_rows  = (M + TILE - 1) / TILE;
    int32_t tile_cols  = (N + TILE - 1) / TILE;
    int     DP_TILE_STRIDE = TILE + 1;
    int     chunk_size = (tile_rows + num_chunks - 1) / num_chunks;

    int64_t hb_off = (int64_t)pair * (tile_rows_max + 1) * bnd_cols;
    int32_t* H_bnd = d_H_bnd + hb_off;

    int32_t* chunk_bnd = d_chunk_bnd + (int64_t)pair * num_chunks * bnd_cols;
    int32_t* delta_bnd = d_delta_bnd + (int64_t)pair * num_chunks * bnd_cols;
    // Each (pair, c) gets its own delta_new slice — no race between blocks
    int32_t* delta_new = d_delta_new  + ((int64_t)pair * (num_chunks-1) + (c-1)) * bnd_cols;

    int tr_start = c * chunk_size;
    int tr_end   = min(tr_start + chunk_size, tile_rows);
    if (tr_start >= tile_rows) {
        if (tx == 0) conv_flags[c] = 1;
        return;
    }

    extern __shared__ char smem[];
    char*    shared_ref = smem;
    char*    shared_qry = shared_ref + TILE;
    int32_t* s_top      = (int32_t*)(shared_qry + TILE + 8);
    int32_t* s_left     = s_top     + (TILE + 1);
    int32_t* s_bot      = s_left    + (TILE + 1);
    int32_t* s_right    = s_bot     + (TILE + 1);
    int32_t* wf_scores  = s_right   + (TILE + 1);
    int32_t* smem_conv  = wf_scores + 3 * (TILE + 1);

    // Paper line 17: s = s[lp] — inject predecessor's final boundary as our top row
    for (int j = tx; j <= N; j += bdx)
        H_bnd[tr_start * bnd_cols + j] = chunk_bnd[(c-1) * bnd_cols + j];
    __syncthreads();

    // Paper lines 18-24: recompute all tile-rows of this chunk
    for (int tr = tr_start; tr < tr_end; ++tr) {
        int row0 = tr * TILE;
        int tM   = min(TILE, M - row0);

        for (int tc = 0; tc < tile_cols; ++tc) {
            int col0 = tc * TILE;
            int tN   = min(TILE, N - col0);

            for (int i = tx; i < tM; i += bdx)
                shared_ref[i] = d_seqs[refStart + row0 + i];
            for (int j = tx; j < tN; j += bdx)
                shared_qry[j] = d_seqs[qryStart + col0 + j];
            for (int idx = tx; idx <= tN; idx += bdx)
                s_top[idx] = H_bnd[tr * bnd_cols + col0 + idx];
            // Left column is always the true NW left edge (tc==0)
            if (tc == 0) {
                for (int idx = tx; idx <= tM; idx += bdx)
                    s_left[idx] = (int32_t)(row0 + idx) * GAP;
            }
            __syncthreads();

            computeTile(tM, tN, s_top, s_left, s_bot, s_right,
                        wf_scores, shared_ref, shared_qry,
                        nullptr, DP_TILE_STRIDE, isProtein, tx, bdx);

            for (int idx = tx; idx <= tN; idx += bdx)
                H_bnd[(tr+1) * bnd_cols + col0 + idx] = s_bot[idx];
            for (int idx = tx; idx <= tM; idx += bdx)
                s_left[idx] = s_right[idx];
            __syncthreads();
        }
    }

    // Update chunk_bnd[c] with this iteration's final output row
    for (int j = tx; j <= N; j += bdx)
        chunk_bnd[c * bnd_cols + j] = H_bnd[tr_end * bnd_cols + j];
    __syncthreads();

    // Compute new delta for this chunk's output boundary
    for (int j = tx; j <= N; j += bdx)
        delta_new[j] = (j == 0)
            ? chunk_bnd[c * bnd_cols]
            : chunk_bnd[c * bnd_cols + j] - chunk_bnd[c * bnd_cols + j - 1];
    __syncthreads();

    // Paper line 21: if (s is parallel to s[i]) → delta vectors match
    // Only check when predecessor is stably converged (conv_flags[c-1] set)
    if (conv_flags[c-1]) {
        if (tx == 0) *smem_conv = 1;
        __syncthreads();

        for (int j = tx; j <= N; j += bdx)
            if (delta_new[j] != delta_bnd[c * bnd_cols + j])
                atomicAnd(smem_conv, 0);
        __syncthreads();

        if (*smem_conv) {
            if (tx == 0) conv_flags[c] = 1;
        }
        __syncthreads();
    }

    // Always update stored delta for next iteration (paper line 24: s[i] = s)
    for (int j = tx; j <= N; j += bdx)
        delta_bnd[c * bnd_cols + j] = delta_new[j];
}

// ==========================================================================
// PHASE 3: Traceback
// Grid = numPairs. One block per pair, sequential traceback.
// H_bnd is fully correct after fix-up converges.
// Left edge replayed from H_bnd (not from V_bnd) for correctness.
// ==========================================================================
__global__ void tracebackPhase(
    const int32_t* __restrict__ d_info,
    const int32_t* __restrict__ d_seqLen,
    const char*    __restrict__ d_seqs,
    uint8_t*                    d_tb,
    const int32_t* __restrict__ d_H_bnd,
    int32_t                     tile_rows_max,
    int32_t                     bnd_cols,
    int32_t*                    d_tile_dp
) {
    const int32_t GAP      = -2;
    const uint8_t DIR_DIAG = 1, DIR_UP = 2, DIR_LEFT = 3;
    int tx  = threadIdx.x;
    int bdx = blockDim.x;

    int32_t numPairs  = d_info[0];
    int32_t maxSeqLen = d_info[1];
    int32_t isProtein = d_info[2];

    int pair = blockIdx.x;
    if (pair >= numPairs) return;

    int32_t M = d_seqLen[2 * pair];
    int32_t N = d_seqLen[2 * pair + 1];

    int32_t refStart = (int64_t)(pair * 2)     * maxSeqLen;
    int32_t qryStart = (int64_t)(pair * 2 + 1) * maxSeqLen;

    int32_t tb_stride      = maxSeqLen * 2 + 2;
    int32_t tbGlobalOffset = (int64_t)pair * tb_stride;
    int     DP_TILE_STRIDE = TILE + 1;

    const int32_t* H_bnd = d_H_bnd + (int64_t)pair * (tile_rows_max + 1) * bnd_cols;
    int32_t*       tile_dp = d_tile_dp + (int64_t)pair * DP_TILE_STRIDE * DP_TILE_STRIDE;

    extern __shared__ char smem[];
    char*    shared_ref = smem;
    char*    shared_qry = shared_ref + TILE;
    int32_t* s_top      = (int32_t*)(shared_qry + TILE + 8);
    int32_t* s_left     = s_top     + (TILE + 1);
    int32_t* s_bot      = s_left    + (TILE + 1);
    int32_t* s_right    = s_bot     + (TILE + 1);
    int32_t* wf_scores  = s_right   + (TILE + 1);

    __shared__ int shared_gi, shared_gj;
    __shared__ int shared_li, shared_lj;
    __shared__ int shared_tr, shared_tc;
    __shared__ int shared_row0, shared_col0;
    __shared__ int shared_tM, shared_tN;
    __shared__ int shared_tbLen;

    if (tx == 0) { shared_gi = M; shared_gj = N; shared_tbLen = 0; }
    __syncthreads();

    while (true) {
        if (tx == 0) {
            while (shared_gi == 0 && shared_gj > 0 && shared_tbLen < tb_stride - 1)
                { d_tb[tbGlobalOffset + shared_tbLen++] = DIR_LEFT; shared_gj--; }
            while (shared_gj == 0 && shared_gi > 0 && shared_tbLen < tb_stride - 1)
                { d_tb[tbGlobalOffset + shared_tbLen++] = DIR_UP;   shared_gi--; }
            if ((shared_gi == 0 && shared_gj == 0) || shared_tbLen >= tb_stride - 1) {
                shared_tr = -1;
            } else {
                shared_tr   = (shared_gi - 1) / TILE;
                shared_tc   = (shared_gj - 1) / TILE;
                shared_row0 = shared_tr * TILE;
                shared_col0 = shared_tc * TILE;
                shared_tM   = min(TILE, M - shared_row0);
                shared_tN   = min(TILE, N - shared_col0);
                shared_li   = shared_gi - shared_row0;
                shared_lj   = shared_gj - shared_col0;
            }
        }
        __syncthreads();
        if (shared_tr < 0) break;

        for (int i = tx; i < shared_tM; i += bdx)
            shared_ref[i] = d_seqs[refStart + shared_row0 + i];

        // Replay tc=0..shared_tc-1 using H_bnd to get correct s_left for this tile-row
        for (int i = tx; i <= shared_tM; i += bdx)
            s_left[i] = (int32_t)(shared_row0 + i) * GAP;
        __syncthreads();

        for (int rtc = 0; rtc < shared_tc; rtc++) {
            int rcol0 = rtc * TILE;
            int rtN   = min(TILE, N - rcol0);
            for (int idx = tx; idx <= rtN; idx += bdx)
                s_top[idx] = H_bnd[shared_tr * bnd_cols + rcol0 + idx];
            for (int j = tx; j < rtN; j += bdx)
                shared_qry[j] = d_seqs[qryStart + rcol0 + j];
            __syncthreads();

            computeTile(shared_tM, rtN, s_top, s_left, s_bot, s_right,
                        wf_scores, shared_ref, shared_qry,
                        nullptr, DP_TILE_STRIDE, isProtein, tx, bdx);

            for (int i = tx; i <= shared_tM; i += bdx)
                s_left[i] = s_right[i];
            __syncthreads();
        }

        for (int j = tx; j <= shared_tN; j += bdx)
            s_top[j] = H_bnd[shared_tr * bnd_cols + shared_col0 + j];
        for (int j = tx; j < shared_tN; j += bdx)
            shared_qry[j] = d_seqs[qryStart + shared_col0 + j];
        __syncthreads();

        computeTile(shared_tM, shared_tN, s_top, s_left, s_bot, s_right,
                    wf_scores, shared_ref, shared_qry,
                    tile_dp, DP_TILE_STRIDE, isProtein, tx, bdx);

        if (tx == 0) {
            while (shared_li >= 1 && shared_lj >= 1 && shared_tbLen < tb_stride - 1) {
                char    r  = shared_ref[shared_li - 1];
                char    q  = shared_qry[shared_lj - 1];
                int32_t sd = tile_dp[(shared_li-1)*DP_TILE_STRIDE+(shared_lj-1)] + subScore(r,q,isProtein);
                int32_t su = tile_dp[(shared_li-1)*DP_TILE_STRIDE+(shared_lj  )] + GAP;
                int32_t sl = tile_dp[(shared_li  )*DP_TILE_STRIDE+(shared_lj-1)] + GAP;
                uint8_t dir  = DIR_DIAG;
                int32_t best = sd;
                if (su > best) { best = su; dir = DIR_UP;   }
                if (sl > best) {            dir = DIR_LEFT; }
                d_tb[tbGlobalOffset + shared_tbLen++] = dir;
                if      (dir == DIR_DIAG) { shared_gi--; shared_gj--; shared_li--; shared_lj--; }
                else if (dir == DIR_UP)   { shared_gi--;               shared_li--;              }
                else                      {               shared_gj--;               shared_lj--; }
            }
        }
        __syncthreads();
    }

    if (tx == 0) {
        int len = shared_tbLen;
        for (int lo = 0, hi = len-1; lo < hi; lo++, hi--) {
            uint8_t tmp                   = d_tb[tbGlobalOffset + lo];
            d_tb[tbGlobalOffset + lo]     = d_tb[tbGlobalOffset + hi];
            d_tb[tbGlobalOffset + hi]     = tmp;
        }
        d_tb[tbGlobalOffset + len] = 0;
    }
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------
void GpuAligner::allocateMem() {
    longestLen = std::max_element(seqs.begin(), seqs.end(),
        [](const Sequence& a, const Sequence& b){ return a.seq.size() < b.seq.size(); }
    )->seq.size();

    auto chk = [](cudaError_t e) {
        if (e != cudaSuccess) { fprintf(stderr,"GPU_ERROR: %s\n",cudaGetErrorString(e)); exit(1); }
    };
    chk(cudaMalloc(&d_seqs,   (size_t)numPairs * 2 * longestLen));
    chk(cudaMalloc(&d_seqLen, (size_t)numPairs * 2 * sizeof(int32_t)));
    chk(cudaMalloc(&d_tb,     (size_t)numPairs * (longestLen * 2 + 2)));
    chk(cudaMalloc(&d_info,   3 * sizeof(int32_t)));
}

void GpuAligner::transferSequence2Device() {
    auto chk = [](cudaError_t e) {
        if (e != cudaSuccess) { fprintf(stderr,"GPU_ERROR: %s\n",cudaGetErrorString(e)); exit(1); }
    };
    std::vector<char> h_seqs((size_t)longestLen * numPairs * 2, 0);
    for (size_t i = 0; i < (size_t)numPairs * 2; ++i)
        std::memcpy(h_seqs.data() + i * longestLen, seqs[i].seq.data(), seqs[i].seq.size());
    chk(cudaMemcpy(d_seqs, h_seqs.data(), (size_t)longestLen * numPairs * 2, cudaMemcpyHostToDevice));

    std::vector<int32_t> h_len(numPairs * 2);
    for (int i = 0; i < numPairs * 2; ++i) h_len[i] = seqs[i].seq.size();
    chk(cudaMemcpy(d_seqLen, h_len.data(), numPairs * 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
    cudaMemset(d_tb, 0, (size_t)numPairs * (longestLen * 2 + 2));

    std::vector<int32_t> h_info = { numPairs, (int32_t)longestLen, isProtein ? 1 : 0 };
    chk(cudaMemcpy(d_info, h_info.data(), 3 * sizeof(int32_t), cudaMemcpyHostToDevice));
}

TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen * 2 + 2;
    TB_PATH h_tb((size_t)tb_length * numPairs);
    cudaMemcpy(h_tb.data(), d_tb, (size_t)tb_length * numPairs, cudaMemcpyDeviceToHost);
    return h_tb;
}

void GpuAligner::alignment() {
    allocateMem();
    transferSequence2Device();
    if (isProtein) initProteinTablesOnce();

    int32_t tile_rows_max = (longestLen + TILE - 1) / TILE;
    int32_t bnd_cols      = longestLen + 1;

    auto chk = [](cudaError_t e, const char* tag) {
        if (e != cudaSuccess) { fprintf(stderr,"GPU_ERROR (%s): %s\n",tag,cudaGetErrorString(e)); exit(1); }
    };

    // H_bnd: full boundary matrix — zeroed so speculative chunks read 0 as top
    size_t hb = (size_t)numPairs * (tile_rows_max + 1) * bnd_cols * sizeof(int32_t);
    int32_t* d_H_bnd = nullptr;
    chk(cudaMalloc(&d_H_bnd, hb),    "H_bnd");
    chk(cudaMemset( d_H_bnd, 0, hb), "H_bnd memset");

    // tile_dp: per-pair tile DP for traceback
    size_t td = (size_t)numPairs * (TILE+1) * (TILE+1) * sizeof(int32_t);
    int32_t* d_tile_dp = nullptr;
    chk(cudaMalloc(&d_tile_dp, td), "tile_dp");

    // chunk_bnd[pair][c][j]: final output boundary of chunk c
    size_t cb = (size_t)numPairs * P * bnd_cols * sizeof(int32_t);
    int32_t* d_chunk_bnd = nullptr;
    chk(cudaMalloc(&d_chunk_bnd, cb),    "chunk_bnd");
    chk(cudaMemset( d_chunk_bnd, 0, cb), "chunk_bnd memset");

    // delta_bnd[pair][c][j]: delta of chunk c's output (for convergence check)
    int32_t* d_delta_bnd = nullptr;
    chk(cudaMalloc(&d_delta_bnd, cb),    "delta_bnd");
    chk(cudaMemset( d_delta_bnd, 0, cb), "delta_bnd memset");

    // delta_new[pair][c-1][j]: scratch for new delta per (pair, chunk) — no races
    size_t dn = (size_t)numPairs * (P-1) * bnd_cols * sizeof(int32_t);
    int32_t* d_delta_new = nullptr;
    chk(cudaMalloc(&d_delta_new, dn),    "delta_new");
    chk(cudaMemset( d_delta_new, 0, dn), "delta_new memset");

    // conv_flags[pair][c]: 1 = chunk c of pair has converged
    size_t cf = (size_t)numPairs * P * sizeof(uint8_t);
    uint8_t* d_conv_flags = nullptr;
    chk(cudaMalloc(&d_conv_flags, cf),    "conv_flags");
    chk(cudaMemset( d_conv_flags, 0, cf), "conv_flags memset");

    // smem: shared_ref+qry+pad + 5*(TILE+1)*int32 + wf_scores[3*(TILE+1)] + smem_conv[1]
    // = (2*128+8) + 7*(129)*4 + 4 = 264 + 3612 + 4 = 3880 bytes
    size_t smem_bytes = (size_t)(2 * TILE + 8)
                      + (size_t)7 * (TILE + 1) * sizeof(int32_t)
                      + sizeof(int32_t);   // smem_conv

    // -----------------------------------------------------------------------
    // Phase 1: all numPairs*P chunks speculatively in parallel
    // -----------------------------------------------------------------------
    forwardPass<<<numPairs * P, 256, smem_bytes>>>(
        d_info, d_seqLen, d_seqs,
        d_H_bnd, tile_rows_max, bnd_cols,
        d_chunk_bnd, d_delta_bnd, P
    );
    chk(cudaGetLastError(),      "phase1 launch");
    chk(cudaDeviceSynchronize(), "phase1 sync");  // ← paper barrier (line 12)

    // Mark chunk 0 as converged for all pairs (it used the true s0)
    {
        std::vector<uint8_t> h_flags(numPairs * P, 0);
        for (int p = 0; p < numPairs; p++) h_flags[p * P] = 1;
        cudaMemcpy(d_conv_flags, h_flags.data(), cf, cudaMemcpyHostToDevice);
    }

    // -----------------------------------------------------------------------
    // Phase 2: RC fix-up — at most P-1 iterations (paper's do-while)
    // -----------------------------------------------------------------------
    std::vector<uint8_t> h_flags(numPairs * P);
    for (int iter = 0; iter < P - 1; ++iter) {
        fixupPhase<<<numPairs * (P-1), 256, smem_bytes>>>(
            d_info, d_seqLen, d_seqs,
            d_H_bnd, tile_rows_max, bnd_cols,
            d_chunk_bnd, d_delta_bnd, d_delta_new,
            d_conv_flags, P
        );
        chk(cudaGetLastError(),      "fixup launch");
        chk(cudaDeviceSynchronize(), "fixup sync");  // ← paper barrier (line 25)

        // Paper line 26: conv = ∧_p conv[p]
        cudaMemcpy(h_flags.data(), d_conv_flags, cf, cudaMemcpyDeviceToHost);
        bool all_conv = true;
        for (int p = 0; p < numPairs && all_conv; p++)
            for (int c = 0; c < P && all_conv; c++)
                if (!h_flags[p * P + c]) all_conv = false;
        if (all_conv) break;
    }

    // -----------------------------------------------------------------------
    // Phase 3: Traceback (sequential per pair, one block per pair)
    // -----------------------------------------------------------------------
    tracebackPhase<<<numPairs, 256, smem_bytes>>>(
        d_info, d_seqLen, d_seqs, d_tb,
        d_H_bnd, tile_rows_max, bnd_cols,
        d_tile_dp
    );
    chk(cudaGetLastError(),      "traceback launch");
    chk(cudaDeviceSynchronize(), "traceback sync");

    TB_PATH tb = transferTB2Host();
    getAlignedSequences(tb);

    cudaFree(d_H_bnd);     cudaFree(d_tile_dp);
    cudaFree(d_chunk_bnd); cudaFree(d_delta_bnd);
    cudaFree(d_delta_new); cudaFree(d_conv_flags);
}

void GpuAligner::getAlignedSequences(TB_PATH& tb_paths) {
    const uint8_t DIR_DIAG = 1, DIR_UP = 2, DIR_LEFT = 3;
    int tb_length = longestLen * 2 + 2;
    tbb::parallel_for(0, numPairs, 1, [&](int pair) {
        int base = tb_length * pair;
        std::string& s0 = seqs[2*pair].seq;
        std::string& s1 = seqs[2*pair+1].seq;
        std::string a0, a1;
        int p0 = 0, p1 = 0;
        for (int i = base; i < base + tb_length; ++i) {
            uint8_t d = tb_paths[i];
            if      (d == DIR_DIAG) { a0 += s0[p0++]; a1 += s1[p1++]; }
            else if (d == DIR_UP)   { a0 += s0[p0++]; a1 += '-';      }
            else if (d == DIR_LEFT) { a0 += '-';       a1 += s1[p1++]; }
            else break;
        }
        seqs[2*pair].aln   = std::move(a0);
        seqs[2*pair+1].aln = std::move(a1);
    });
}

void GpuAligner::clearAndReset() {
    cudaFree(d_seqs); cudaFree(d_seqLen); cudaFree(d_tb); cudaFree(d_info);
    seqs.clear(); longestLen = 0; numPairs = 0;
}

void GpuAligner::writeAlignment(std::string fileName, bool append) {
    std::ofstream f;
    if (append) f.open(fileName, std::ios::app);
    else        f.open(fileName);
    if (!f) { fprintf(stderr,"ERROR: can't open %s\n", fileName.c_str()); exit(1); }
    for (auto& seq : seqs)
        f << '>' << seq.name << '\n' << seq.aln << '\n';
}
