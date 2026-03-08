// -----------------------------------------------------------
// alignment.cu — Tiled Needleman-Wunsch (tentative) baseline
//
// Architecture:
//   Forward pass:  TILE×TILE tiles, row-by-row, left-to-right.
//                  Wavefront parallelism (BLOCKSIZE=256 threads, can be changed) within each tile.
//                  Stores H_bnd (bottom row) and V_bnd (right column) per tile.
//
//   Traceback:     Walks (M,N)->(0,0). For each tile on the path:
//                  1. Look up boundaries from H_bnd + V_bnd
//                  2. Recompute tile DP (BLOCKSIZE=256 threads together, wavefront)
//                  3. Thread 0 walks within the tile with the stored scores
//
// Memory layout:
//   H_bnd[pair][tr+1][j]            — score at row (tr+1)*TILE, column j
//   V_bnd[pair][tr*tile_cols+tc][i] — score at row row0+i, column (tc+1)*TILE
//   tile_dp[pair][i][j]             — scratch for one tile during traceback
// -----------------------------------------------------------
#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <tbb/parallel_for.h>

#define TILE 512

// Protein scoring: constant memory tables
__device__ __constant__ int8_t d_aa_to_idx[256];
__device__ __constant__ int8_t d_blosum62[20*20];

// Instantiate BLOSUM62 matrix into global device memory
static void initProteinTablesOnce() {
    static bool inited = false;
    if (inited) return;
    inited = true;

    // ASCII -> index in "ARNDCQEGHILKMFPSTWYV"
    int8_t aa_to_idx[256];
    for (int i = 0; i < 256; i++) aa_to_idx[i] = -1;

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

/**
 * Helper function to calculate substituion score.
 * MATCH/MISMATCH for DNA, BLOSUM62 matrix for Protein amino acids
 */
__device__ __forceinline__ int16_t subScore(
    char r, char q,
    int32_t isProtein
) {
    const int16_t MATCH = 2;
    const int16_t MISMATCH = -1;

    if (!isProtein) {
        return (r == q) ? MATCH : MISMATCH;
    }

    int8_t ri = d_aa_to_idx[(unsigned char)r];
    int8_t qi = d_aa_to_idx[(unsigned char)q];
    
    // Find score for ref and query AAs in the matrix, use MISMATCH in the worst case the AA is not in the matrix
    return (ri >= 0 && qi >= 0) ? (int16_t)d_blosum62[ri * 20 + qi] : MISMATCH; 
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


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#include <cuda/ptx>
#endif

__device__ __forceinline__ void max_score(
    int16_t score_diag, int16_t score_up, int16_t score_left, int16_t &score_out,
    uint8_t &dir_out, uint8_t DIR_DIAG, uint8_t DIR_UP, uint8_t DIR_LEFT
) {
// have to add __CUDA_ARCH__ >= 900 to avoid compilation error on A30
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)    
    uint32_t du = score_diag | (score_up << 16);

    // put UP into lane0 (also lane1, but lane1 doesn't matter)
    uint32_t c  = score_up | (score_up << 16);

    // lane0: max(diag,up), lane1: max(up,up)=up
    int32_t out = __viaddmax_s16x2(0, du, c);

    int16_t best = out & 0xFFFF;
    // dir for diag vs up
    uint8_t dir = (score_up > score_diag) ? DIR_UP : DIR_DIAG;

    if (score_left > best) {
        best = score_left;
        dir = DIR_LEFT;
    }
    score_out = best;
    dir_out = dir;

#else // same as the original code
    score_out = score_diag;
    dir_out = DIR_DIAG;
    if (score_up > score_out) {
        score_out = score_up;
        dir_out = DIR_UP;
    }
    if (score_left > score_out) {
        score_out = score_left;
        dir_out = DIR_LEFT;
    }
#endif
}



/**
 * CUDA Kernel: Full tiled NW global alignment on the GPU. One block per pair.
 *
 * Data structures:
 * - H_bnd[tr+1][col] : bottom boundary scores after finishing tile-row tr.
 *
 * - V_bnd[tr][tc][i] : optimization for traceback(!!!).
 *   Stores the RIGHT boundary of tile (tr,tc), i in [0..tM].
 *   During TRACEBACK, left boundary of tile(tc) is right boundary of tile(tc-1),
 *   so we can get it in O(TILE) loads, no need to rerunning previous tiles.
 *
 * - tile_dp : per-pair buffer (TILE+1)^2.
 *   Used only during traceback to store the current tile score matrix
 *   so thread0 can walk inside the tile.
 */


__global__ void alignmentOnGPU (
    int32_t* d_info,       // [0]: numPairs, [1]: maxSeqLen, [2]: isProtein
    int32_t* d_seqLen,     // Array of sequence lengths
    char* d_seqs,          // Flat array of sequences
    uint8_t* d_tb,         // Output traceback paths
    int16_t* d_H_bnd,      // horizontal boundaries [pair][tile_row+1][col]
    int32_t  tile_rows_max,
    int32_t  bnd_cols,     // maxSeqLen + 1
    int16_t* d_V_bnd,      // vertical boundaries [pair][tr*tile_cols_max+tc][TILE+1]
    int32_t  tile_cols_max,
    int16_t* d_tile_dp     // per pair tile DP buffer [(TILE+1)^2]
) {
    // -----------------------------------------------------------
    // KERNEL CONFIGURATION
    // -----------------------------------------------------------
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    
    // Scoring Scheme (MATCH/MISMATCH moved to helper function subScore)
    const int16_t GAP = -2;

    // Traceback Direction Constants (DO NOT MODIFY)
    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    int32_t numPairs  = d_info[0];
    int32_t maxSeqLen = d_info[1];
    int32_t isProtein = d_info[2];

    int pair = bx;
    if (pair >= numPairs) return;

    int32_t M = d_seqLen[2 * pair];      // ref len
    int32_t N = d_seqLen[2 * pair + 1];  // qry len
    if (M > maxSeqLen || N > maxSeqLen) return;

    // Calculate memory offsets for this pair
    int32_t refStart = (pair * 2) * maxSeqLen;
    int32_t qryStart = (pair * 2 + 1) * maxSeqLen;

    // number of tiles for this pair
    int32_t tile_rows = (M + TILE - 1) / TILE;
    int32_t tile_cols = (N + TILE - 1) / TILE;

    int32_t tb_stride = maxSeqLen * 2 + 2;
    int32_t tbGlobalOffset = pair * tb_stride; 

    // pointers into global boundary arrays (int64 to avoid overflow!)
    int64_t hb_pair_offset = (int64_t)pair * (tile_rows_max + 1) * bnd_cols;
    int16_t* H_bnd = d_H_bnd + hb_pair_offset;

    // pointer to this pair's tile DP buffer
    int DP_TILE_STRIDE = TILE + 1;
    int16_t* tile_dp = d_tile_dp + (int64_t)pair * DP_TILE_STRIDE * DP_TILE_STRIDE;

    // V_bnd base for this pair
    int64_t vb_pair_offset = (int64_t)pair * tile_rows_max * tile_cols_max * (TILE + 1);
    int16_t* V_bnd = d_V_bnd + vb_pair_offset;


    // ---------------------------------------------------
    // Shared memory pointers
    // ---------------------------------------------------
    extern __shared__ char smem[];
    char* shared_ref = smem;
    char* shared_qry = shared_ref + TILE;
    int16_t* s_top = (int16_t*)(shared_qry + TILE);
    int16_t* s_left = s_top + (TILE + 1);
    int16_t* s_bot = s_left + (TILE + 1);
    int16_t* s_right = s_bot + (TILE + 1);
    int16_t* wf_scores = s_right + (TILE + 1);

    const int WF_STRIDE = TILE + 1;


    // ---------------------------------------------------
    // FORWARD PASS: fill all tiles, store H_bnd and V_bnd
    //
    // Within each tile: anti-diagonal wavefront (k = i+j),
    // i.e. cell(i,j) needs cell(i-1,j-1), cell(i-1,j), cell(i,j-1).
    // 256(BLOCKSIZE) threads process cells on the same diagonal in parallel.
    // ---------------------------------------------------


    // NW base case: top row of DP matrix 
    for (int j = tx; j <= N; j += blockDim.x) {
        H_bnd[j] = (int16_t)(j * GAP);
    }
    __syncthreads();

    for (int tr = 0; tr < tile_rows; ++tr) {
        int row0 = tr * TILE;
        // actual rows in this tile
        int tM = min(TILE, M - row0);

        for (int tc = 0; tc < tile_cols; ++tc) {
            int col0 = tc * TILE;
            // actual cols in this tile
            int tN = min(TILE, N - col0);

            // Load tile's sequences into shared memory
            // Using memory coalescing
            for (int i = tx; i < tM; i += blockDim.x) shared_ref[i] = d_seqs[refStart + row0 + i];
            for (int j = tx; j < tN; j += blockDim.x) shared_qry[j] = d_seqs[qryStart + col0 + j];

            // load top boundary for columns in this tile
            for (int idx = tx; idx <= tN; idx += blockDim.x) s_top[idx] = H_bnd[tr * bnd_cols + col0 + idx];

            // left boundary: NW base case for first tile-column, else carried from previous tile
            if (tc == 0) {
                for (int idx = tx; idx <= tM; idx += blockDim.x){
                    s_left[idx] = (int16_t)(row0 + idx) * GAP;
                }
            }

            // init wf scratch, s_bot, s_right
            for (int idx = tx; idx < 3 * WF_STRIDE; idx += blockDim.x) wf_scores[idx] = (int16_t)-9999;
            for (int idx = tx; idx <= tM; idx += blockDim.x) s_right[idx] = (int16_t)-9999;
            for (int idx = tx; idx <= tN; idx += blockDim.x) s_bot[idx] = (int16_t)-9999;

            __syncthreads();

            // anti-diagonal wavefront:
            for (int k = 0; k <= tM + tN; ++k) {
                // Cyclic buffers for 3-wavefront dependency
                int curr_k   = (k % 3) * WF_STRIDE;
                int pre_k    = ((k + 2) % 3) * WF_STRIDE;
                int prepre_k = ((k + 1) % 3) * WF_STRIDE;

                // TODO: should we clear current diagonal buffer?
                for (int idx = tx; idx < WF_STRIDE; idx += blockDim.x) {
                    wf_scores[curr_k + idx] = (int16_t)-9999;
                }
                __syncthreads();

                // Compute loop bounds for this diagonal
                int i_start = max(0, k - tN);
                int i_end   = min(tM, k);
                // Wavefront: each thread handles cells on this diagonal with stride
                for (int i = i_start + tx; i <= i_end; i += blockDim.x) {
                    int j = k - i;

                    int16_t score = -9999;
                    uint8_t direction = DIR_DIAG;

                    // -- Boundary Conditions --
                    if (i == 0 && j == 0) {
                        score = s_top[0];
                    }
                    else if (i == 0) {
                        score = s_top[j];
                    }
                    else if (j == 0) {
                        score = s_left[i];
                    } 
                    else {
                        // -- Inner Cell Calculation --
                        char r_char = shared_ref[i - 1];
                        char q_char = shared_qry[j - 1];
                        
                        // diag neighbor (i-1, j-1): diagonal k-2
                        int16_t diag_base;
                        if      (i - 1 == 0 && j - 1 == 0) diag_base = s_top[0];
                        else if (i - 1 == 0)               diag_base = s_top[j - 1];
                        else if (j - 1 == 0)               diag_base = s_left[i - 1];
                        else                               diag_base = wf_scores[prepre_k + (i - 1)];

                        // substitution score: BLOSUM62 for protein, MATCH/MISMATCH for DNA
                        int16_t sub = subScore(r_char, q_char, isProtein);
                        int16_t score_diag = diag_base + sub;

                        // up neighbor (i-1, j): diagonal k-1
                        int16_t score_up;
                        if (i - 1 == 0)
                            score_up = s_top[j];
                        else
                            score_up = wf_scores[pre_k + (i - 1)];
                        score_up += GAP;

                        // left neighbor (i, j-1): diagonal k-1
                        int16_t score_left;
                        if (j - 1 == 0)
                            score_left = s_left[i];
                        else
                            score_left = wf_scores[pre_k + i];
                        score_left += GAP;

                        max_score(score_diag, score_up, score_left, score, direction, DIR_DIAG, DIR_UP, DIR_LEFT);
                    }
                    // write Score
                    wf_scores[curr_k + i] = score;

                    // write to right/bottom boundary if on tile edge
                    if (j == tN) s_right[i] = score;
                    if (i == tM) s_bot[j] = score;

                }
                __syncthreads();  // barrier between diagonals
            } // End Wavefront Loop


            // store boundaries to global memory for traceback
            // H_bnd[tr+1]: bottom boundary (as top boundary for tile-row tr+1)
            for (int idx = tx; idx <= tN; idx += blockDim.x)
                H_bnd[(tr + 1) * bnd_cols + col0 + idx] = s_bot[idx];

            // V_bnd[tr][tc]: right boundary (as left boundary for traceback of tile tc+1)
            int64_t vb_tile_base = (tr * tile_cols_max + tc) * (TILE + 1);
            for (int idx = tx; idx <= tM; idx += blockDim.x)
                V_bnd[vb_tile_base + idx] = s_right[idx];

            // pass right boundary as left boundary to the next tile in this row
            for (int idx = tx; idx <= tM; idx += blockDim.x)
                s_left[idx] = s_right[idx];

            __syncthreads();

        } // end tile_cols loop

        }// end tile_rows loop

    // ---------------------------------------------------
    // TRACEBACK & REVERSAL
    
    // For each tile on the path:
    //   1. Load boundaries: H_bnd for top, V_bnd for left
    //   2. Recompute full tile DP together (256 threads, wavefront),
    //      and stores scores in tile_dp[] in global memory
    //   3. Thread 0 walks within the tile using tile_dp scores to recover
    //      directions on the fly (so no need for direction storage)
    // ---------------------------------------------------
    __shared__ int shared_gi, shared_gj;          // current global DP coords
    __shared__ int shared_li, shared_lj;          // current local coordinates within tile
    __shared__ int shared_tr, shared_tc;          // current tile indices
    __shared__ int shared_row0, shared_col0;      // tile origin in global coords
    __shared__ int shared_tM, shared_tN;          // actual tile sizes
    __shared__ int shared_tbLen;                  // traceback path len so far

    if (tx == 0) {
        shared_gi = M;
        shared_gj = N;
        shared_tbLen = 0;
    }
    __syncthreads();

    while (true) {
        // thread0 handles boundary (row0/col0)
        if (tx == 0) {
            while (shared_gi == 0 && shared_gj > 0 && shared_tbLen < tb_stride - 1) {
                    d_tb[tbGlobalOffset + shared_tbLen++] = DIR_LEFT;
                    shared_gj--;
            }
            while (shared_gj == 0 && shared_gi > 0 && shared_tbLen < tb_stride - 1) {
                    d_tb[tbGlobalOffset + shared_tbLen++] = DIR_UP;
                    shared_gi--;
            }
            if ((shared_gi == 0 && shared_gj == 0) || shared_tbLen >= tb_stride - 1) {
                shared_tr = -1; // traceback done
            } else {
                // determine which tile we are in
                shared_tr = (shared_gi - 1) / TILE;
                shared_tc = (shared_gj - 1) / TILE;
                shared_row0 = shared_tr * TILE;
                shared_col0 = shared_tc * TILE;
                shared_tM = min(TILE, M - shared_row0);
                shared_tN = min(TILE, N - shared_col0);
                shared_li = shared_gi - shared_row0;
                shared_lj = shared_gj - shared_col0;
            }
        }
        __syncthreads();
        if (shared_tr < 0) break;


        /// 1. Load boundaries for this tile
        for (int i = tx; i < shared_tM; i += blockDim.x)
            shared_ref[i] = d_seqs[refStart + shared_row0 + i];
        for (int j = tx; j < shared_tN; j += blockDim.x)
            shared_qry[j] = d_seqs[qryStart + shared_col0 + j];
        // load top boundary from H_bnd
        for (int j = tx; j <= shared_tN; j += blockDim.x)
            s_top[j] = H_bnd[shared_tr * bnd_cols + shared_col0 + j];
        // left boundary for this tile: lookup V_bnd 
        if (shared_tc == 0) {
            for (int i = tx; i <= shared_tM; i += blockDim.x)
                s_left[i] = (int16_t)(shared_row0 + i) * GAP;
        } else {
            // left boundary = right boundary of the tile (tr, tc-1)
            int64_t vb_left_tile_base = (shared_tr * tile_cols_max + (shared_tc - 1)) * (TILE + 1);
            for (int i = tx; i <= shared_tM; i += blockDim.x)
                s_left[i] = V_bnd[vb_left_tile_base + i];
        }
        __syncthreads();

        // 2. Recompute tile DP  (wavefront parallelism, same as forward)
        // write boundaries into tile_dp
        for (int i = tx; i <= shared_tM; i += blockDim.x)
            tile_dp[i * DP_TILE_STRIDE + 0] = s_left[i];
        for (int j = tx; j <= shared_tN; j += blockDim.x)
            tile_dp[0 * DP_TILE_STRIDE + j] = s_top[j];

        for (int idx = tx; idx < 3 * WF_STRIDE; idx += blockDim.x)
            wf_scores[idx] = (int16_t)-9999;
        __syncthreads();

        for (int k = 0; k <= shared_tM + shared_tN; ++k) {
            int curr_k   = (k % 3) * WF_STRIDE;
            int pre_k    = ((k + 2) % 3) * WF_STRIDE;
            int prepre_k = ((k + 1) % 3) * WF_STRIDE;

            for (int idx = tx; idx < WF_STRIDE; idx += blockDim.x)
                wf_scores[curr_k + idx] = (int16_t)-9999;
            __syncthreads();

            int i_start = max(0, k - shared_tN);
            int i_end   = min(shared_tM, k);

            for (int i = i_start + tx; i <= i_end; i += blockDim.x) {
                int j = k - i;

                int16_t score;
                uint8_t dummy_dir = DIR_DIAG;

                if (i == 0 && j == 0) score = s_top[0];
                else if (i == 0)      score = s_top[j];
                else if (j == 0)      score = s_left[i];
                else {
                    char r_char = shared_ref[i - 1];
                    char q_char = shared_qry[j - 1];

                    int16_t diag_base;
                    if      (i - 1 == 0 && j - 1 == 0) diag_base = s_top[0];
                    else if (i - 1 == 0)               diag_base = s_top[j - 1];
                    else if (j - 1 == 0)               diag_base = s_left[i - 1];
                    else                               diag_base = wf_scores[prepre_k + (i - 1)];

                    int16_t sub = subScore(r_char, q_char, isProtein);
                    int16_t score_diag = diag_base + sub;

                    int16_t score_up;
                    if (i - 1 == 0) score_up = s_top[j];
                    else            score_up = wf_scores[pre_k + (i - 1)];
                    score_up += GAP;

                    int16_t score_left;
                    if (j - 1 == 0) score_left = s_left[i];
                    else            score_left = wf_scores[pre_k + i];
                    score_left += GAP;

                    max_score(score_diag, score_up, score_left, score, dummy_dir, DIR_DIAG, DIR_UP, DIR_LEFT);
                }

                wf_scores[curr_k + i] = score;
                // store full matrix for thread0 traceback
                tile_dp[i * DP_TILE_STRIDE + j] = score;
            }
            __syncthreads();
        }
        // 3. Thread 0 walks within the tile using tile_dp scores
        if (tx == 0) {
            while (shared_li >= 1 && shared_lj >= 1 && shared_tbLen < tb_stride - 1) {
                char r = shared_ref[shared_li - 1];
                char q = shared_qry[shared_lj - 1];

                int16_t sd = tile_dp[(shared_li - 1) * DP_TILE_STRIDE + (shared_lj - 1)] + subScore(r, q, isProtein);
                int16_t su = tile_dp[(shared_li - 1) * DP_TILE_STRIDE + (shared_lj)] + GAP;
                int16_t sl = tile_dp[(shared_li) * DP_TILE_STRIDE + (shared_lj - 1)] + GAP;

                uint8_t dir = DIR_DIAG;
                int16_t best = sd;
                if (su > best) {
                    best = su;
                    dir = DIR_UP;
                }
                if (sl > best) {
                    best = sl;
                    dir = DIR_LEFT;
                }
                d_tb[tbGlobalOffset + shared_tbLen++] = dir;
                
                // Move coordinates
                if (dir == DIR_DIAG) { shared_gi--; shared_gj--; shared_li--; shared_lj--; }
                else if (dir == DIR_UP) { shared_gi--; shared_li--; }
                else { shared_gj--; shared_lj--; }
            }
        }
        __syncthreads();
    }


    // reverse tb for this pair
    if (tx == 0) {
        // reverse in-place
        int localLen = shared_tbLen;
        for (int lo = 0, hi = localLen - 1; lo < hi; lo++, hi--) {
            uint8_t tmp = d_tb[tbGlobalOffset + lo];
            d_tb[tbGlobalOffset + lo] = d_tb[tbGlobalOffset + hi];
            d_tb[tbGlobalOffset + hi] = tmp;
        }
        d_tb[tbGlobalOffset + localLen] = 0;

    }
    
}


void GpuAligner::alignment() {

    // TODO: make sure to appropriately set the values below
    int numBlocks = numPairs;  // i.e. number of thread blocks on the GPU
    int blockSize = 256; // i.e. number of GPU threads per thread block

    // 1. Allocate memory on Device
    allocateMem();
    
    // 2. Transfer sequence to device
    transferSequence2Device();

    // Upload BLOSUM62 / aa_to_idx to constant memory if this is a protein alignment
    if (isProtein) initProteinTablesOnce();

    int32_t tile_rows_max = (longestLen + TILE - 1) / TILE;
    int32_t tile_cols_max = (longestLen + TILE - 1) / TILE;
    int32_t bnd_cols = longestLen + 1;

    // 3. Allocate H_bnd: bottom boundary at each tile-row
    size_t hb_bytes = (size_t)numPairs * (tile_rows_max + 1) * bnd_cols * sizeof(int16_t);
    int16_t* d_H_bnd = nullptr;
    cudaError_t err = cudaMalloc(&d_H_bnd, hb_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_H_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemset(d_H_bnd, 0, hb_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (memset d_H_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    // 4. Allocate V_bnd: right boundary at each tile
    size_t vb_bytes = (size_t)numPairs * tile_rows_max * tile_cols_max * (TILE + 1) * sizeof(int16_t);
    int16_t* d_V_bnd = nullptr;
    err = cudaMalloc(&d_V_bnd, vb_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_V_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemset(d_V_bnd, 0, vb_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (memset d_V_bnd): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // 5. Allocate tile_dp: buffer for traceback within a tile for a pair
    size_t tile_dp_stride = (size_t)(TILE + 1) * (TILE + 1);
    size_t tile_dp_bytes = (size_t)numPairs * tile_dp_stride * sizeof(int16_t);
    int16_t* d_tile_dp = nullptr;
    err = cudaMalloc(&d_tile_dp, tile_dp_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_tile_dp): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Shared memory: sequences + 4 boundaries + 3-slot wavefront
    size_t smem_bytes = 2 * TILE                          // shared_ref, shared_qry
                      + 4 * (TILE + 1) * sizeof(int16_t)  // s_top, s_left, s_bot, s_right
                      + 3 * (TILE + 1) * sizeof(int16_t); // wf_scores

    // 6. Perform the alignment on GPU
    alignmentOnGPU<<<numBlocks, blockSize, smem_bytes>>>(d_info, d_seqLen, d_seqs, d_tb, d_H_bnd, tile_rows_max, bnd_cols,d_V_bnd, tile_cols_max, d_tile_dp);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (launch): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (runtime): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // 7. Transfer the traceback path from device
    TB_PATH tb_paths = transferTB2Host();
    cudaDeviceSynchronize();

    // 8. Get the aligned sequence with traceback paths
    getAlignedSequences(tb_paths);

    // 9. Free
    cudaFree(d_H_bnd);
    cudaFree(d_V_bnd);
    cudaFree(d_tile_dp);


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

void GpuAligner::clearAndReset () {
    cudaFree(d_seqs);
    cudaFree(d_seqLen);
    cudaFree(d_tb);
    cudaFree(d_info);
    seqs.clear();
    longestLen = 0;
    numPairs = 0;

}

/**
 * Writes the aligned sequences to a file in FASTA format.
 * Each sequence is written with a header line ('>' + name) followed by the aligned sequence.
 * If `append` is true, the output is appended to the file; otherwise, the file is overwritten.
 */
void GpuAligner::writeAlignment(std::string fileName, bool append) {
    std::ofstream outFile;
    if (append) outFile.open(fileName, std::ios::app);
    else        outFile.open(fileName);
    if (!outFile) {
        fprintf(stderr, "ERROR: cant open file: %s\n", fileName.c_str());
        exit(1);
    }
    for (auto& seq: seqs) {
        outFile << ('>' + seq.name + '\n');
        outFile << (seq.aln + '\n');
    }
    outFile.close();
}
