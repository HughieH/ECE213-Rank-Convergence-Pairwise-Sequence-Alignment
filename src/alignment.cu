
// ==========================================================================
// Tiled NW with Rank Convergence (3-kernel approach)
//
// RC kernels:
//     forwardPass -> Parallel Forward Pass kernel: 
//                    numPairs * P blocks,
//                    block = pair * P + c owns chunk c
//     fixupPhase -> Fix-up kernel: 
//                   numPairs * (P-1) blocks, iterates <= P-1 times
//     tracebackPhase -> Traceback kernel: 
//                       numPairs blocks (similar to baseline)
//
// TILE = 128 keeps shared memory at 3.9KB per block
// nz (speculative start non-zero vector) = all +1 as required by paper 
// 
// Changes:
//  1. Fixup block runs only when predecessor is converged and self not converged:
//     (this avoids launching useful work on chunks that cannot converge yet)
//
//  2. Remove d_delta_new scratch (big global memory traffic) and compute deltas on-the-fly.
//     Still updates delta_bnd[c] each iteration.
//
//  3. Add device flag d_any_work: if an iteration does no recompute at all, host breaks early.
// ==========================================================================

#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <tbb/parallel_for.h>

// Tile dimension: each tile covers a TILE x TILE subregion of the M x N DP matrix
#define TILE 512
// Number of RC chunks ("processors" run in parallel as mentioned in paper)
#define P 4

// Protein scoring tables in constant memory
__device__ __constant__ int8_t d_aa_to_idx[256];   // ASCII -> BLOSUM62 row/col index
__device__ __constant__ int8_t d_blosum62[20*20];  

static void initProteinTablesOnce() {
    static bool inited = false;
    if (inited) return;
    inited = true;

    int8_t aa_to_idx[256];
    for (int i = 0; i < 256; i++) {
        aa_to_idx[i] = -1;
    }
    const char* AA = "ARNDCQEGHILKMFPSTWYV";
    for (int i = 0; i < 20; i++) {
        aa_to_idx[(unsigned char)AA[i]] = (int8_t)i;
    }

    // BLOSUM62 in ordering: A R N D C Q E G H I L K M F P S T W Y V
    static const int8_t blosum62[20 * 20] = {
         4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0, // A
        -1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3, // R
        -2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, // N
        -2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, // D
         0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1, // C
        -1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, // Q
        -1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, // E
         0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3, // G
        -2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, // H
        -1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3, // I
        -1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1, // L
        -1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, // K
        -1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1, // M
        -2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1, // F
        -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2, // P
         1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, // S
         0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0, // T
        -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3, // W
        -2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1, // Y
         0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4  // V
    };

    cudaMemcpyToSymbol(d_aa_to_idx, aa_to_idx, sizeof(aa_to_idx));
    cudaMemcpyToSymbol(d_blosum62, blosum62, sizeof(blosum62));
}

// Returns substitution score for reference and query char
__device__ __forceinline__ int32_t subScore(char r, char q, int32_t isProtein) {
    const int32_t MATCH = 2;
    const int32_t MISMATCH = -1;

    if (!isProtein) return (r == q) ? MATCH : MISMATCH;

    int8_t ri = d_aa_to_idx[(unsigned char)r];
    int8_t qi = d_aa_to_idx[(unsigned char)q];

    return (ri >= 0 && qi >= 0) ? (int32_t)d_blosum62[ri * 20 + qi] : MISMATCH;
}

// ==========================================================================
// computeTile: helper function that fills one TILE x TILE submatrix of the NW DP table.
// Basically the same as our baseline implementation.
// ==========================================================================
__device__ void computeTile(
    int tM, int tN, // actual number of rows (M) and cols (N) in this tile
    int32_t* __restrict__ s_top, // top boundary scores [0..tN]
    int32_t* __restrict__ s_left, // left boundary scores [0..tM]
    int32_t* __restrict__ s_bot, // bottom boundary scores [0..tN]
    int32_t* __restrict__ s_right, // right boundary scores [0..tM]
    int32_t* __restrict__ wf_scores, // rolling 3-diagonal wavefront buffer
    char* __restrict__ shared_ref, // reference chars for this tile-row
    char* __restrict__ shared_qry, // query chars for this tile-col
    int32_t* __restrict__ tile_dp, // full DP scores written here (traceback only, else nullptr)
    int DP_TILE_STRIDE, // row stride of tile_dp (= TILE+1)
    int32_t isProtein, // 1 = BLOSUM62, 0 = match/mismatch
    int tx, int bdx
) {
    const int32_t NEG_INF = -1000000; // value for uninitialized cells
    const int32_t GAP = -2;           // linear gap penalty
    const int WFS = TILE + 1;         // wavefront buffer width
    // const uint8_t DIR_DIAG = 1, DIR_UP = 2, DIR_LEFT = 3;

    for (int idx = tx; idx < 3*WFS; idx += bdx) {
        wf_scores[idx] = NEG_INF;
    }
    for (int idx = tx; idx <= tM; idx += bdx) {
        s_right[idx] = NEG_INF;
    }
    for (int idx = tx; idx <= tN; idx += bdx) {
        s_bot[idx] = NEG_INF;
    }

    // if storing full DP (traceback), copy boundary values into tile_dp edges
    if (tile_dp) {
        for (int i = tx; i <= tM; i += bdx) {
            tile_dp[i * DP_TILE_STRIDE] = s_left[i];
        }
        for (int j = tx; j <= tN; j += bdx) {
            tile_dp[0 * DP_TILE_STRIDE + j] = s_top[j];
        }
    }
    __syncthreads();

    // process each anti-diagonal k = i + j in order (wavefront sweep)
    for (int k = 0; k <= tM + tN; ++k) {
        int curr_k = (k % 3) * WFS;       // current diagonal's slot
        int pre_k = ((k+2) % 3) * WFS;    // k-1 diagonal (UP/LEFT gap source)
        int prepre_k = ((k+1) % 3) * WFS; // k-2 diagonal (DIAG score source)

        for (int idx = tx; idx < WFS; idx += bdx) {
            wf_scores[curr_k + idx] = NEG_INF;
        }
        __syncthreads();

        // all cells (i, j) on diagonal k where i+j=k, clamped to tile bounds
        int i_start = max(0, k - tN);
        int i_end = min(tM, k);

        for (int i = i_start + tx; i <= i_end; i += bdx) {
            int j = k - i;
            int32_t score;
            // uint8_t dir = DIR_DIAG;

            if (i == 0 && j == 0) {
                // top-left corner: inherit from s_top[0] (= s_left[0] by NW base case)
                score = s_top[0];
            } else if (i == 0) {
                // top edge: value comes entirely from the top boundary row
                score = s_top[j];
            } else if (j == 0) {
                // left edge: value comes entirely from the left boundary column
                score = s_left[i];
            } else {
                // general cell: pick best of diagonal, up (gap in query), left (gap in ref)
                int32_t diag_base =
                    (i-1==0 && j-1==0) ? s_top[0] : // corner of boundary
                    (i-1==0) ? s_top[j-1] : // diagonal from top boundary
                    (j-1==0) ? s_left[i-1] : // diagonal from left boundary
                    wf_scores[prepre_k+(i-1)]; // diagonal from wavefront

                int32_t sd = diag_base + subScore(shared_ref[i-1], shared_qry[j-1], isProtein);

                int32_t su = ((i-1==0) ? s_top[j]  : wf_scores[pre_k+(i-1)]) + GAP; // gap in ref
                int32_t sl = ((j-1==0) ? s_left[i] : wf_scores[pre_k+i])     + GAP; // gap in qry

                score = sd;
                // dir = DIR_DIAG;
                if (su > score) {
                    score = su;
                    // dir = DIR_UP;
                }
                if (sl > score) {
                    score = sl;
                    // dir = DIR_LEFT;
                }
            }

            wf_scores[curr_k + i] = score;
            if (j == tN) {
                s_right[i] = score; // reached right edge, write to s_right
            }
            if (i == tM) {
                s_bot[j] = score;   // reached bottom edge, write to s_bot
            }
            if (tile_dp) {
                tile_dp[i * DP_TILE_STRIDE + j] = score;
            }
        }
        __syncthreads();
    }
}

// ==========================================================================
// PHASE 1: forwardPass — Speculative parallel forward pass
// All P chunks of all pairs run simultaneously.
//
// Grid:  numPairs * P blocks
// Block: pair * P + c  -> owns chunk c of sequence pair
//
// Chunk 0: uses true NW boundaries (s0 in paper).
// Chunks 1..P-1: speculative, nz=+1 left col, top row=0 (zeroed by cudaMemset).
//
// After this kernel, chunk_bnd[c] holds the bottom boundary of chunk c,
// and delta_bnd[c] holds its delta representation for convergence checking.
// ==========================================================================
__global__ void forwardPass(
    const int32_t* __restrict__ d_info, // [numPairs, maxSeqLen, isProtein]
    const int32_t* __restrict__ d_seqLen, // device array of sequence lengths
    const char* __restrict__ d_seqs, // device array of packed sequences
    int32_t* d_H_bnd, // horizontal boundary matrix
    int32_t tile_rows_max, // max number of tile-rows across all pairs
    int32_t bnd_cols, // width of each boundary row (= maxSeqLen + 1)
    int32_t* d_chunk_bnd, // snapshot of each chunk's final output boundary (per pair)
    int32_t* d_delta_bnd, // delta representation of chunk_bnd for convergence (per pair)
    int32_t num_chunks // number of RC chunks (equal to P)
) {
    const int32_t GAP = -2;

    int tx = threadIdx.x;
    int bdx = blockDim.x;

    int32_t numPairs = d_info[0];
    int32_t maxSeqLen = d_info[1];
    int32_t isProtein = d_info[2];

    // decode which pair and chunk this block owns
    int pair = blockIdx.x / num_chunks;
    int c = blockIdx.x % num_chunks;
    if (pair >= numPairs) return;

    int32_t M = d_seqLen[2 * pair];     // reference sequence length
    int32_t N = d_seqLen[2 * pair + 1]; // query sequence length

    // byte offsets into the packed sequence array
    int32_t refStart = (int64_t)(pair * 2) * maxSeqLen;
    int32_t qryStart = (int64_t)(pair * 2 + 1) * maxSeqLen;

    int32_t tile_rows = (M + TILE - 1) / TILE; // number of tile-rows for this pair
    int32_t tile_cols = (N + TILE - 1) / TILE; // number of tile-cols for this pair
    int DP_TILE_STRIDE = TILE + 1;

    // each chunk owns an equal slice of tile-rows
    int chunk_size = (tile_rows + num_chunks - 1) / num_chunks;

    // pointer to this pair's horizontal boundary submatrix in d_H_bnd
    int64_t hb_off = (int64_t)pair * (tile_rows_max + 1) * bnd_cols;
    int32_t* H_bnd = d_H_bnd + hb_off;

    // pointers into per-pair chunk boundary and delta arrays
    int32_t* chunk_bnd = d_chunk_bnd + (int64_t)pair * num_chunks * bnd_cols;
    int32_t* delta_bnd = d_delta_bnd + (int64_t)pair * num_chunks * bnd_cols;

    // tile-row range owned by this chunk
    int tr_start = c * chunk_size;
    int tr_end = min(tr_start + chunk_size, tile_rows);
    if (tr_start >= tile_rows) return; // chunk out of range for short sequences

    // ---------------------------------------------------
    // Shared memory pointers, used in computeTile
    // ---------------------------------------------------
    extern __shared__ char smem[];
    char* shared_ref = smem;
    char* shared_qry = shared_ref + TILE;
    int32_t* s_top = (int32_t*)(shared_qry + TILE + 8);
    int32_t* s_left = s_top + (TILE + 1);
    int32_t* s_bot = s_left + (TILE + 1);
    int32_t* s_right = s_bot + (TILE + 1);
    int32_t* wf_scores = s_right + (TILE + 1);

    // chunk 0 initializes the true NW base row
    if (c == 0) {
        for (int j = tx; j <= N; j += bdx) {
            H_bnd[j] = (int32_t)j * GAP;
        }
        __syncthreads();
    }
    // chunks 1..P-1 read H_bnd[tr_start * bnd_cols + j] which is 0 (from cudaMemset).
    // this is the speculative nz top boundary — incorrect but non-(-inf)

    // sweep all tile-rows in this chunk, left to right across tile-cols
    for (int tr = tr_start; tr < tr_end; ++tr) {
       
        int row0 = tr * TILE;           // first row index of this tile-row
        int tM = min(TILE, M - row0);   // actual rows in this tile

        for (int tc = 0; tc < tile_cols; ++tc) {
            int col0 = tc * TILE;           // first col index of this tile-col
            int tN = min(TILE, N - col0);   // actual cols in this tile

            // load reference and query characters into shared memory
            for (int i = tx; i < tM; i += bdx) {
                shared_ref[i] = d_seqs[refStart + row0 + i];
            }
            for (int j = tx; j < tN; j += bdx) {
                shared_qry[j] = d_seqs[qryStart + col0 + j];
            }

            // top boundary: row tr of H_bnd (written by the previous tile-row's s_bot)
            for (int idx = tx; idx <= tN; idx += bdx) {
                s_top[idx] = H_bnd[tr * bnd_cols + col0 + idx];
            }

            // left boundary: only set at tc==0 (left edge of this tile-row)
            if (tc == 0) {
                if (c == 0) {
                    // true NW left column
                    for (int idx = tx; idx <= tM; idx += bdx) {
                        s_left[idx] = (int32_t)(row0 + idx) * GAP;
                    }
                } else {
                    // speculative nz = +1
                    for (int idx = tx; idx <= tM; idx += bdx) {
                        s_left[idx] = 1;
                    }
                }
            }
            // for tc > 0, s_left was set to s_right at the end of the previous tc iteration
            __syncthreads();

            computeTile(tM, tN, s_top, s_left, s_bot, s_right,
                        wf_scores, shared_ref, shared_qry,
                        nullptr, DP_TILE_STRIDE, isProtein, tx, bdx);

            // write bottom boundary into H_bnd for the next tile-row to read as its top
            for (int idx = tx; idx <= tN; idx += bdx) {
                H_bnd[(tr+1) * bnd_cols + col0 + idx] = s_bot[idx];
            }
            // carry s_right forward as s_left for the next tile-col
            for (int idx = tx; idx <= tM; idx += bdx) {
                s_left[idx] = s_right[idx];
            }
            __syncthreads();
        }
    }

    // snapshot the final bottom boundary of this chunk into chunk_bnd[c]
    for (int j = tx; j <= N; j += bdx) {
        chunk_bnd[c * bnd_cols + j] = H_bnd[tr_end * bnd_cols + j];
    }
    __syncthreads();

    // compute and store the delta representation of chunk_bnd[c]
    // delta[j] = chunk_bnd[j] - chunk_bnd[j-1] (difference of adjacent boundary values).
    // two boundary vectors are "parallel" (rank-1 convergence) if their deltas are equal.
    for (int j = tx; j <= N; j += bdx) {
        delta_bnd[c * bnd_cols + j] = (j == 0)
            ? chunk_bnd[c * bnd_cols]
            : chunk_bnd[c * bnd_cols + j] - chunk_bnd[c * bnd_cols + j - 1];
    }
}

// ==========================================================================
// PHASE 2: fixupPhase — RC fix-up iteration
//
// (do-while fix-up loop body, parallel.for p in 2..P).
// All P-1 non-zero chunks of all pairs run simultaneously in each iteration.
// The host calls this kernel at most P-1 times, stopping early if all converged.
// Each cudaDeviceSynchronize between launches acts as the paper's barrier (line 25).
//
// Grid:  numPairs * (P-1) blocks
// Block: pair * (P-1) + (c-1)  ->  owns chunk c of a pair
// 
// UPDATES:
// only runs if predecessor converged and self not converged
// removes d_delta_new: compute delta on the fly
// sets d_any_work=1 if any block did real recompute this iteration
// ==========================================================================
__global__ void fixupPhase(
    const int32_t* __restrict__ d_info,
    const int32_t* __restrict__ d_seqLen,
    const char* __restrict__ d_seqs,
    int32_t* d_H_bnd,
    int32_t tile_rows_max,
    int32_t bnd_cols, // width of each boundary row (= maxSeqLen + 1)
    int32_t* d_chunk_bnd, // per-chunk final output boundary [P * bnd_cols] per pair
    int32_t* d_delta_bnd, // delta of chunk_bnd [P * bnd_cols] per pair
    uint8_t* d_conv_flags, // conv_flags[c]=1 means chunk c has converged and need not be rerun
    int32_t num_chunks, // number of RC chunks (= P)
    int32_t* d_any_work
) {
    const int32_t GAP = -2;
    int tx = threadIdx.x;
    int bdx = blockDim.x;

    int32_t numPairs = d_info[0];
    int32_t maxSeqLen = d_info[1];
    int32_t isProtein = d_info[2];

    // decode which pair and chunk this block owns
    int pair = blockIdx.x / (num_chunks - 1);
    int c = blockIdx.x % (num_chunks - 1) + 1; // c in range [1, P-1]
    if (pair >= numPairs) return;

    uint8_t* conv_flags = d_conv_flags + pair * num_chunks;

    // early exit: this chunk and its predecessor are both already converged
    if (conv_flags[c] || !conv_flags[c - 1]) return;

    // mark that some work happens in this iteration
    if (tx == 0) {
        atomicExch(d_any_work, 1);
    }

    int32_t M = d_seqLen[2 * pair];
    int32_t N = d_seqLen[2 * pair + 1];

    int32_t refStart = (int64_t)(pair * 2) * maxSeqLen;
    int32_t qryStart = (int64_t)(pair * 2 + 1) * maxSeqLen;

    int32_t tile_rows = (M + TILE - 1) / TILE;
    int32_t tile_cols = (N + TILE - 1) / TILE;

    int DP_TILE_STRIDE = TILE + 1;
    int chunk_size = (tile_rows + num_chunks - 1) / num_chunks;

    int64_t hb_off = (int64_t)pair * (tile_rows_max + 1) * bnd_cols;
    int32_t* H_bnd = d_H_bnd + hb_off;

    int32_t* chunk_bnd = d_chunk_bnd + (int64_t)pair * num_chunks * bnd_cols;
    int32_t* delta_bnd = d_delta_bnd + (int64_t)pair * num_chunks * bnd_cols;

    int tr_start = c * chunk_size;
    int tr_end = min(tr_start + chunk_size, tile_rows);

    if (tr_start >= tile_rows) {
                // chunk has no tile-rows (sequence shorter than expected) — mark converged
        if (tx == 0) {
            conv_flags[c] = 1;
        }
        return;
    }

    // ---------------------------------------------------
    // Shared memory pointers, used in computeTile
    // ---------------------------------------------------
    extern __shared__ char smem[];
    char* shared_ref = smem;
    char* shared_qry = shared_ref + TILE;
    int32_t* s_top = (int32_t*)(shared_qry + TILE + 8);
    int32_t* s_left = s_top + (TILE + 1);
    int32_t* s_bot = s_left + (TILE + 1);
    int32_t* s_right = s_bot + (TILE + 1);
    int32_t* wf_scores = s_right + (TILE + 1);
    int32_t* smem_conv = wf_scores + 3 * (TILE + 1); // single int for convergence vote

    // inject predecessor's latest output boundary as our top row
    // chunk_bnd[c-1] is the corrected (or most-recently-updated) bottom of chunk c-1
    for (int j = tx; j <= N; j += bdx) {
        H_bnd[tr_start * bnd_cols + j] = chunk_bnd[(c-1) * bnd_cols + j];
    }    
    __syncthreads();

    // recompute this chunk
    for (int tr = tr_start; tr < tr_end; ++tr) {
        int row0 = tr * TILE;
        int tM = min(TILE, M - row0);

        for (int tc = 0; tc < tile_cols; ++tc) {
            int col0 = tc * TILE;
            int tN = min(TILE, N - col0);

            for (int i = tx; i < tM; i += bdx) {
                shared_ref[i] = d_seqs[refStart + row0 + i];
            }
            for (int j = tx; j < tN; j += bdx) {
                shared_qry[j] = d_seqs[qryStart + col0 + j];
            }
            for (int idx = tx; idx <= tN; idx += bdx) {
                s_top[idx] = H_bnd[tr * bnd_cols + col0 + idx];
            }
            // left column is always the true NW global left edge: s[i,0] = i * GAP
            if (tc == 0) {
                for (int idx = tx; idx <= tM; idx += bdx) {
                    s_left[idx] = (int32_t)(row0 + idx) * GAP;
                }
            }
            __syncthreads();

            computeTile(tM, tN, s_top, s_left, s_bot, s_right,
                        wf_scores, shared_ref, shared_qry,
                        nullptr, DP_TILE_STRIDE, isProtein, tx, bdx);

            for (int idx = tx; idx <= tN; idx += bdx) {
                H_bnd[(tr+1) * bnd_cols + col0 + idx] = s_bot[idx];
            }
            for (int idx = tx; idx <= tM; idx += bdx) {
                s_left[idx] = s_right[idx];
            }
            __syncthreads();
        }
    }

    // update chunk_bnd[c] with this iteration's recomputed final output row
    for (int j = tx; j <= N; j += bdx) {
        chunk_bnd[c * bnd_cols + j] = H_bnd[tr_end * bnd_cols + j];
    }
    __syncthreads();

    // compare delta(chunk_bnd[c]) vs delta_bnd[c] (previous)
    if (tx == 0) {
        *smem_conv = 1; // assume converged
    }
    __syncthreads();

    for (int j = tx; j <= N; j += bdx) {
        int32_t cur = chunk_bnd[c * bnd_cols + j];
        int32_t delta = (j == 0) ? cur : (cur - chunk_bnd[c * bnd_cols + (j - 1)]);
        int32_t prev_delta = delta_bnd[c * bnd_cols + j];
        if (delta != prev_delta) {
            atomicAnd(smem_conv, 0); // any mismatch means not converged
        }
    }
    __syncthreads();

    if (*smem_conv) {
        // deltas match, this chunk's output is now parallel to the correct boundary
        if (tx == 0) {
            conv_flags[c] = 1;
        }
    }
    __syncthreads();

    // update delta_bnd[c] for next iteration
    for (int j = tx; j <= N; j += bdx) {
        int32_t cur = chunk_bnd[c * bnd_cols + j];
        int32_t delta = (j == 0) ? cur : (cur - chunk_bnd[c * bnd_cols + (j - 1)]);
        delta_bnd[c * bnd_cols + j] = delta;
    }
}

// ==========================================================================
// PHASE 3: tracebackPhase — Reconstruct alignment from converged DP boundaries
//
// Near identical to baseline implementation
//
// For each tile on the path:
//   1. Load boundaries: H_bnd for top, V_bnd for left
//   2. Recompute full tile DP together (256 threads, wavefront),
//      and stores scores in tile_dp[] in global memory
//   3. Thread 0 walks within the tile using tile_dp scores to recover
//      directions on the fly (so no need for direction storage)
//
// Grid:  numPairs blocks (one block per pair)
// Block: blockIdx.x == pair index
// ==========================================================================
__global__ void tracebackPhase(
    const int32_t* __restrict__ d_info,
    const int32_t* __restrict__ d_seqLen,
    const char* __restrict__ d_seqs,
    uint8_t* d_tb, // output traceback path array [(maxSeqLen*2+2)] per pair
    const int32_t* __restrict__ d_H_bnd,
    int32_t tile_rows_max,
    int32_t bnd_cols,
    int32_t* d_tile_dp // scratch space [(TILE+1)*(TILE+1)] per pair for the current tile's DP scores
) {
    const int32_t GAP = -2;
    const uint8_t DIR_DIAG = 1, DIR_UP = 2, DIR_LEFT = 3;
    int tx = threadIdx.x;
    int bdx = blockDim.x;

    int32_t numPairs = d_info[0];
    int32_t maxSeqLen = d_info[1];
    int32_t isProtein = d_info[2];

    int pair = blockIdx.x;
    if (pair >= numPairs) return;

    int32_t M = d_seqLen[2 * pair];
    int32_t N = d_seqLen[2 * pair + 1];

    int32_t refStart = (int64_t)(pair * 2) * maxSeqLen;
    int32_t qryStart = (int64_t)(pair * 2 + 1) * maxSeqLen;

    int32_t tb_stride = maxSeqLen * 2 + 2;             // max traceback path length
    int32_t tbGlobalOffset = (int64_t)pair * tb_stride; // byte offset into d_tb
    int DP_TILE_STRIDE = TILE + 1;

    // pointers to this pair's data
    const int32_t* H_bnd = d_H_bnd + (int64_t)pair * (tile_rows_max + 1) * bnd_cols;
    int32_t* tile_dp = d_tile_dp + (int64_t)pair * DP_TILE_STRIDE * DP_TILE_STRIDE;

    // ---------------------------------------------------
    // Shared memory pointers, used in computeTile
    // ---------------------------------------------------
    extern __shared__ char smem[];
    char* shared_ref = smem;
    char* shared_qry = shared_ref + TILE;
    int32_t* s_top = (int32_t*)(shared_qry + TILE + 8);
    int32_t* s_left = s_top + (TILE + 1);
    int32_t* s_bot = s_left + (TILE + 1);
    int32_t* s_right = s_bot + (TILE + 1);
    int32_t* wf_scores = s_right + (TILE + 1);

    // shared state for the single-threaded traceback pointer (only tx==0 advances it)
    __shared__ int shared_gi, shared_gj;     // current global (row, col) in full DP matrix
    __shared__ int shared_li, shared_lj;     // local (row, col) within the current tile
    __shared__ int shared_tr, shared_tc;     // current tile indices
    __shared__ int shared_row0, shared_col0; // tile origin in global coords
    __shared__ int shared_tM, shared_tN;     // actual tile sizes
    __shared__ int shared_tbLen;             // traceback path len so far

    if (tx == 0) {
        shared_gi = M;
        shared_gj = N;
        shared_tbLen = 0;
    }
    __syncthreads();

    while (true) {
        // thread0 handles boundary (row0/col0)
        if (tx == 0) {
            // consume remaining gaps along the top row (gi==0)
            while (shared_gi == 0 && shared_gj > 0 && shared_tbLen < tb_stride - 1) {
                d_tb[tbGlobalOffset + shared_tbLen++] = DIR_LEFT;
                shared_gj--;
            }
            // consume remaining gaps along the left column (gj==0)
            while (shared_gj == 0 && shared_gi > 0 && shared_tbLen < tb_stride - 1) {
                d_tb[tbGlobalOffset + shared_tbLen++] = DIR_UP;
                shared_gi--;
            }
            if ((shared_gi == 0 && shared_gj == 0) || shared_tbLen >= tb_stride - 1) {
                shared_tr = -1; // traceback done!
            } else {
                shared_tr = (shared_gi - 1) / TILE;
                shared_tc = (shared_gj - 1) / TILE;
                shared_row0 = shared_tr * TILE;
                shared_col0 = shared_tc * TILE;
                shared_tM = min(TILE, M - shared_row0);
                shared_tN = min(TILE, N - shared_col0);
                shared_li = shared_gi - shared_row0; // local row within tile
                shared_lj = shared_gj - shared_col0; // local col within tile
            }
        }
        
        __syncthreads();
        if (shared_tr < 0) break;

        // all threads cooperatively load the reference characters for this tile-row
        for (int i = tx; i < shared_tM; i += bdx) {
            shared_ref[i] = d_seqs[refStart + shared_row0 + i];
        }

        // reconstruct s_left for the target tile-col by replaying tile-cols 0..tc-1.
        // we cannot read V_bnd (it was never stored) so we recompute from H_bnd.
        for (int i = tx; i <= shared_tM; i += bdx) {
            s_left[i] = (int32_t)(shared_row0 + i) * GAP; // true NW left edge as starting point
        }
        __syncthreads();

        for (int rtc = 0; rtc < shared_tc; rtc++) {
            int rcol0 = rtc * TILE;
            int rtN = min(TILE, N - rcol0);
            for (int idx = tx; idx <= rtN; idx += bdx) {
                s_top[idx] = H_bnd[shared_tr * bnd_cols + rcol0 + idx];
            }
            for (int j = tx; j < rtN; j += bdx) {
                shared_qry[j] = d_seqs[qryStart + rcol0 + j];
            }
            __syncthreads();

            // replay this tile to advance s_left -> s_right, then carry forward
            computeTile(shared_tM, rtN, s_top, s_left, s_bot, s_right,
                        wf_scores, shared_ref, shared_qry,
                        nullptr, DP_TILE_STRIDE, isProtein, tx, bdx);

            for (int i = tx; i <= shared_tM; i += bdx) {
                s_left[i] = s_right[i];
            }
            __syncthreads();
        }

        // load the actual traceback tile's top and query, then compute with tile_dp
        for (int j = tx; j <= shared_tN; j += bdx) {
            s_top[j] = H_bnd[shared_tr * bnd_cols + shared_col0 + j];
        }
        for (int j = tx; j < shared_tN; j += bdx) {
            shared_qry[j] = d_seqs[qryStart + shared_col0 + j];
        }
        __syncthreads();

        // recompute with tile_dp to capture full DP scores for direction recovery
        computeTile(shared_tM, shared_tN, s_top, s_left, s_bot, s_right,
                    wf_scores, shared_ref, shared_qry,
                    tile_dp, DP_TILE_STRIDE, isProtein, tx, bdx);

        // tx==0 walks the traceback within this tile until it exits a boundary
        if (tx == 0) {
            while (shared_li >= 1 && shared_lj >= 1 && shared_tbLen < tb_stride - 1) {
                char r = shared_ref[shared_li - 1];
                char q = shared_qry[shared_lj - 1];
                // recover direction by re-evaluating the three NW recurrence cases
                int32_t sd = tile_dp[(shared_li-1)*DP_TILE_STRIDE+(shared_lj-1)] + subScore(r, q, isProtein);
                int32_t su = tile_dp[(shared_li-1)*DP_TILE_STRIDE+(shared_lj  )] + GAP;
                int32_t sl = tile_dp[(shared_li  )*DP_TILE_STRIDE+(shared_lj-1)] + GAP;
                uint8_t dir = DIR_DIAG;
                int32_t best = sd;
                if (su > best) {
                    best = su;
                    dir = DIR_UP;
                }
                if (sl > best) {
                    dir = DIR_LEFT;
                }
                d_tb[tbGlobalOffset + shared_tbLen++] = dir;
                if (dir == DIR_DIAG) {
                    shared_gi--;
                    shared_gj--;
                    shared_li--;
                    shared_lj--;
                } else if (dir == DIR_UP) {
                    shared_gi--;
                    shared_li--;
                } else {
                    shared_gj--;
                    shared_lj--;
                }
            }
        }
        __syncthreads();
    }

    // reverse tb for this pair
    if (tx == 0) {
        // reverse in-place
        int len = shared_tbLen;
        for (int lo = 0, hi = len-1; lo < hi; lo++, hi--) {
            uint8_t tmp = d_tb[tbGlobalOffset + lo];
            d_tb[tbGlobalOffset + lo] = d_tb[tbGlobalOffset + hi];
            d_tb[tbGlobalOffset + hi] = tmp;
        }
        d_tb[tbGlobalOffset + len] = 0; 
    }
}


/**
 * Allocates GPU memory for sequences, lengths, and traceback paths.
 * Calculates 'longestLen' to determine the stride for flattening the sequence array.
 */
void GpuAligner::allocateMem() {
    // 1. Find the maximum sequence length to determine memory stride
    longestLen = std::max_element(seqs.begin(), seqs.end(), [](const Sequence& a, const Sequence& b) {
        return a.seq.size() < b.seq.size();
    })->seq.size();

    cudaError_t err;
    
    // 2. Allocate flat array for all sequences (Reference + Query pairs)
    // Layout: [Seq0_Ref ...pad... | Seq0_Qry ...pad... | Seq1_Ref ... ]
    err = cudaMalloc(&d_seqs, (size_t)numPairs * 2 * longestLen * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 3. Allocate array for sequence lengths (to handle padding correctly)
    err = cudaMalloc(&d_seqLen, numPairs * 2 * sizeof(int32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 4. Allocate Traceback Buffer
    // Worst case path is roughly 2x sequence length (all gaps)
    int tb_length = longestLen * 2 + 2; 
    err = cudaMalloc(&d_tb, (size_t)numPairs * tb_length * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 5. Allocate mem for device info, 3 ints: numPairs, longestLen, isProtein
    err = cudaMalloc(&d_info, 3 * sizeof(int32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
}


/**
 * Flattens the host sequence objects into a single 1D array and transfers to GPU.
 */
void GpuAligner::transferSequence2Device() {
    cudaError_t err;
    
    // 1. Flatten sequences on Host
    // We use a fixed stride 'longestLen' to simplify indexing on the GPU
    std::vector<char> h_seqs((size_t)longestLen * numPairs * 2, 0); 
    
    for (size_t i = 0; i < (size_t)numPairs * 2; ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(h_seqs.data() + (i * longestLen), s.data(), s.size());
    }

    // 2. Transfer flattened sequences to Device
    err = cudaMemcpy(d_seqs, h_seqs.data(), (size_t)longestLen * numPairs * 2 * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 3. Transfer sequence lengths to Device
    std::vector<int32_t> h_seqLen(numPairs * 2, 0);
    for (int i = 0; i < numPairs * 2; ++i) h_seqLen[i] = seqs[i].seq.size();
    
    err = cudaMemcpy(d_seqLen, h_seqLen.data(), numPairs * 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 4. Initialize Traceback buffer on Device (Zero out)
    int tb_length = longestLen * 2 + 2;
    err = cudaMemset(d_tb, 0, (size_t)tb_length * numPairs * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 5. Transfer Meta Info
    std::vector<int32_t> h_info(3);
    h_info[0] = numPairs;
    h_info[1] = longestLen;
    h_info[2] = isProtein ? 1: 0; // Pass in 1 if isProtein flag is enabled in main entrypoint

    err = cudaMemcpy(d_info, h_info.data(), 3 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
}


/**
 * Copies the computed traceback paths from GPU back to Host.
 */
TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen * 2 + 2;
    TB_PATH h_tb((size_t)tb_length * numPairs);

    cudaError_t err = cudaMemcpy(h_tb.data(), d_tb, (size_t)tb_length * numPairs * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
    return h_tb;
}


void GpuAligner::alignment() {

    // NOTE: numBlocks removed, defind in the kernel calls themselves now
    int blockSize = 256; // i.e. number of GPU threads per thread block

    // 1. Allocate memory on Device
    allocateMem();

    // 2. Transfer sequence to device
    transferSequence2Device();

    // Upload BLOSUM62 / aa_to_idx to constant memory if this is a protein alignment
    if (isProtein) initProteinTablesOnce();

    int32_t tile_rows_max = (longestLen + TILE - 1) / TILE;
    int32_t bnd_cols = longestLen + 1;

    // 3. H_bnd: full horizontal boundary matrix — zeroed so speculative chunks read 0 as their top
    size_t hb = (size_t)numPairs * (tile_rows_max + 1) * bnd_cols * sizeof(int32_t);
    int32_t* d_H_bnd = nullptr;
    cudaError_t err = cudaMalloc(&d_H_bnd, hb);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_H_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemset(d_H_bnd, 0, hb);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (memset d_H_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // 4. Allocate tile_dp: buffer for traceback within a tile for a pair
    size_t td = (size_t)numPairs * (TILE+1) * (TILE+1) * sizeof(int32_t);
    int32_t* d_tile_dp = nullptr;
    err = cudaMalloc(&d_tile_dp, td);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_tile_dp): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    // 5. Allocate chunk_bnd: snapshot of each chunk's final output boundary row [P * bnd_cols] per pair
    size_t cb = (size_t)numPairs * P * bnd_cols * sizeof(int32_t);
    int32_t* d_chunk_bnd = nullptr;
    err = cudaMalloc(&d_chunk_bnd, cb);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_chunk_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }
     err = cudaMemset(d_chunk_bnd, 0, cb);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (memset d_chunk_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // 6. Allocate delta_bnd: delta representation of chunk_bnd for convergence checking [P * bnd_cols] per pair
    int32_t* d_delta_bnd = nullptr;
    err = cudaMalloc(&d_delta_bnd, cb);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_delta_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }
     err = cudaMemset(d_delta_bnd, 0, cb);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (memset d_delta_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }


    // 8. Allocate conv_flags: per-chunk convergence flags [P] per pair; 1 = converged, 0 = needs another fixup
    size_t cf = (size_t)numPairs * P * sizeof(uint8_t);
    uint8_t* d_conv_flags = nullptr;
    // chk(cudaMalloc(&d_conv_flags, cf), "conv_flags");
    // chk(cudaMemset(d_conv_flags, 0, cf), "conv_flags memset");
    err = cudaMalloc(&d_conv_flags, cf);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_conv_flags): %s\n", cudaGetErrorString(err));
        exit(1);
    }
     err = cudaMemset(d_conv_flags, 0, cf);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (memset d_conv_flags): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // any-work flag for fixup early break
    int32_t* d_any_work = nullptr;
    err = cudaMalloc(&d_any_work, sizeof(int32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_any_work): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // shared memory per block: ref+qry+pad + 5 boundary arrays + wf_scores + smem_conv
    size_t smem_bytes = (size_t)(2 * TILE + 8)
                      + (size_t)7 * (TILE + 1) * sizeof(int32_t)
                      + sizeof(int32_t);

    // -----------------------------------------------------------------------
    // Phase 1: all numPairs*P chunks speculatively in parallel 
    // -----------------------------------------------------------------------
    forwardPass<<<numPairs * P, blockSize, smem_bytes>>>(
        d_info, d_seqLen, d_seqs,
        d_H_bnd, tile_rows_max, bnd_cols,
        d_chunk_bnd, d_delta_bnd, P
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (forwardPass launch): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (forwardPass sync): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // chunk 0 used the true s0 -> it is correct by definition, mark it converged
    {
        std::vector<uint8_t> h_flags(numPairs * P, 0);
        for (int p = 0; p < numPairs; p++) {
            h_flags[p * P] = 1;
        }
        cudaMemcpy(d_conv_flags, h_flags.data(), cf, cudaMemcpyHostToDevice);
    }

    // -----------------------------------------------------------------------
    // Phase 2: RC fix-up — at most P-1 kernel launches 
    // -----------------------------------------------------------------------
    std::vector<uint8_t> h_flags(numPairs * P);
    for (int iter = 0; iter < P - 1; ++iter) {
        int32_t zero = 0;
        cudaMemcpy(d_any_work, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);

        fixupPhase<<<numPairs * (P-1), blockSize, smem_bytes>>>(
            d_info, d_seqLen, d_seqs,
            d_H_bnd, tile_rows_max, bnd_cols,
            d_chunk_bnd, d_delta_bnd,
            d_conv_flags, P, d_any_work
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR (fixupPhase launch): %s\n", cudaGetErrorString(err));
            exit(1);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "GPU_ERROR (fixupPhase sync): %s\n", cudaGetErrorString(err));
            exit(1);
        }

        // if no blocks did real work, break
        int32_t h_any = 0;
        cudaMemcpy(&h_any, d_any_work, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (h_any == 0) break;

        // keep all converged break
        // check if all chunks of all pairs converged
        cudaMemcpy(h_flags.data(), d_conv_flags, cf, cudaMemcpyDeviceToHost);
        bool all_conv = true;
        for (int p = 0; p < numPairs && all_conv; p++) {
            for (int c = 0; c < P && all_conv; c++) {
                if (!h_flags[p * P + c]) {
                    all_conv = false;
                }
            }
        }
        if (all_conv) break; // early exit — best case is just 1 fix-up iteration
    }

    // -----------------------------------------------------------------------
    // Phase 3: Traceback — one block per pair, sequential within each block
    // -----------------------------------------------------------------------    
    tracebackPhase<<<numPairs, blockSize, smem_bytes>>>(
        d_info, d_seqLen, d_seqs, d_tb,
        d_H_bnd, tile_rows_max, bnd_cols,
        d_tile_dp
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (tracebackPhase launch): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (tracebackPhase sync): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Transfer the traceback path from device
    TB_PATH tb = transferTB2Host();

    // Get the aligned sequence with traceback paths
    getAlignedSequences(tb);

    // Free mem
    cudaFree(d_H_bnd);
    cudaFree(d_tile_dp);
    cudaFree(d_chunk_bnd);
    cudaFree(d_delta_bnd);
    cudaFree(d_conv_flags);
}


void GpuAligner::clearAndReset () {
    cudaFree(d_seqs);
    cudaFree(d_seqLen);
    cudaFree(d_tb);
    cudaFree(d_info);
    seqs.clear();
    longestLen = 0;
    numPairs = 0;

}


// ---------------------------------------------------
//  The following functions are the same as PA2
// ---------------------------------------------------

/** * Reconstructs the actual string alignment from the traceback paths (CIGAR-like data).
 * Converts directional codes (DIAG, UP, LEFT) into aligned strings with gaps.
 */
void GpuAligner::getAlignedSequences (TB_PATH& tb_paths) {

    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;
    
    int tb_length = longestLen * 2 + 2;
    
    // TODO: Apply parallelism to this for loop
    // HINT: Remember to add the header
    tbb::parallel_for(0, numPairs, 1, [&] (int pair) {
        int tb_start = tb_length * pair;
        
        int seqId0 = 2 * pair;
        int seqId1 = 2 * pair + 1;
        std::string& seq0 = seqs[seqId0].seq;
        std::string& seq1 = seqs[seqId1].seq;
        std::string aln0 = "";
        std::string aln1 = "";
        int seqPos0 = 0;
        int seqPos1 = 0;

        // Iterate through the recorded path directions
        for (int i = tb_start; i < tb_start + tb_length; ++i) {
            if (tb_paths[i] == DIR_DIAG) {
                // Match/Mismatch
                aln0 += seq0[seqPos0];
                aln1 += seq1[seqPos1];
                seqPos0++; seqPos1++;
            }
            else if (tb_paths[i] == DIR_UP) {
                // Deletions (gap on seq1)
                aln0 += seq0[seqPos0];
                aln1 += '-';
                seqPos0++;
            }
            else if (tb_paths[i] == DIR_LEFT) {
                // Insertions (gap on seq0)
                aln0 += '-';
                aln1 += seq1[seqPos1];
                seqPos1++;
            }
            else {
                // End of the tb_path (encountered 0 or uninitialized value)
                break;
            }
        }

        // Save results
        seqs[seqId0].aln = std::move(aln0);
        seqs[seqId1].aln = std::move(aln1);
    });
}


/**
 * Writes the aligned sequences to a file in FASTA format.
 * Each sequence is written with a header line ('>' + name) followed by the aligned sequence.
 * If `append` is true, the output is appended to the file; otherwise, the file is overwritten.
 */
void GpuAligner::writeAlignment(std::string fileName, bool append) {
    std::ofstream f;
    if (append) {
        f.open(fileName, std::ios::app);
    } else {
        f.open(fileName);
    }
    if (!f) { fprintf(stderr, "ERROR: can't open %s\n", fileName.c_str()); exit(1); }
    for (auto& seq : seqs) {
        f << '>' << seq.name << '\n' << seq.aln << '\n';
    }
}