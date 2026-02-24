#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <tbb/parallel_for.h>

// GACT Parameters, moved to global scope
const int T = 185;        // Tile size
const int O = 40;         // Overlap between tiles

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
    err = cudaMalloc(&d_seqs, numPairs * 2 * longestLen * sizeof(char));
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
    int tb_length = longestLen << 1; 
    err = cudaMalloc(&d_tb, numPairs * tb_length * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 5. Allocate meta-info struct (numPairs, maxLen)
    err = cudaMalloc(&d_info, 2 * sizeof(int32_t));
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
    std::vector<char> h_seqs(longestLen * numPairs * 2, 0); 
    
    for (size_t i = 0; i < numPairs * 2; ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(h_seqs.data() + (i * longestLen), s.data(), s.size());
    }

    // 2. Transfer flattened sequences to Device
    err = cudaMemcpy(d_seqs, h_seqs.data(), longestLen * numPairs * 2 * sizeof(char), cudaMemcpyHostToDevice);
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
    int tb_length = longestLen << 1;
    std::vector<uint8_t> h_tb (tb_length * numPairs, 0);
    
    err = cudaMemcpy(d_tb, h_tb.data(), tb_length * numPairs * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 5. Transfer Meta Info
    std::vector<int32_t> h_info (2);
    h_info[0] = numPairs;
    h_info[1] = longestLen;
    err = cudaMemcpy(d_info, h_info.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
}

/**
 * Copies the computed traceback paths from GPU back to Host.
 */
TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen << 1;
    TB_PATH h_tb(tb_length * numPairs);

    cudaError_t err = cudaMemcpy(h_tb.data(), d_tb, tb_length * numPairs * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
    return h_tb;
}

/**
 * CUDA Kernel: Performs tiled alignment (GACT) on the GPU.
 * (DONE) Optimize using shared memory, wavefront parallelism, and memory coalescing.
 * HINT: 
 * Consider
 * 1. Number of threads for each step (initialization, filling scoring matrix, traceback)
 * 2. Which variables should go in registers vs shared memory
 * 3. Where should __syncthreads() be added?
 * 4. TODOs marked below are the main tasks, but other parts may also need changes for correctness or efficiency
 * 5. You may add/modify shared memory, registers, or helper functions as needed, as long as output is valid
 */
__global__ void alignmentOnGPU (
    int32_t* d_info,       // [0]: numPairs, [1]: maxSeqLen
    int32_t* d_seqLen,     // Array of sequence lengths
    char* d_seqs,          // Flat array of sequences
    uint8_t* d_tb          // Output traceback paths
) {
    // -----------------------------------------------------------
    // KERNEL CONFIGURATION
    // -----------------------------------------------------------
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    
    // Scoring Scheme (DO NOT MODIFY)
    const int16_t MATCH = 2;
    const int16_t MISMATCH = -1;
    const int16_t GAP = -2;

    // Traceback Direction Constants (DO NOT MODIFY)
    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    // Shared Memory Allocation
    // tbDir: Stores direction for every cell in the tile (T x T)
    // Note: We only store directions for the inner matrix (indices 1..T)
    __shared__ uint8_t tbDir[T * T]; 
    
    // wf_scores: 3 arrays (Current, Previous, Pre-Previous wavefronts) 
    // needed for diagonal computation. Size T+1 to include boundary 0.
    __shared__ int16_t wf_scores[3 * (T + 1)]; 

    // localPath: Temporary buffer to store tile traceback (reversed)
    __shared__ uint8_t localPath[2 * T];

    // State variables
    __shared__ bool lastTile;
    __shared__ int16_t maxScore;
    __shared__ int32_t best_ti;
    __shared__ int32_t best_tj;

    // -----------------------------------------------------------
    // BASELINE IMPLEMENTATION - Not Parallilized! Left here for reference
    // -----------------------------------------------------------

    // HINT: Use shared memory to store the reference and query segments involved in the tile
    // __shared__ char shared_ref[??];
    // __shared__ char shared_qry[??];

    // if (bx == 0 && tx == 0) { // We will want to get rid of this probably, unless we need it for initializing?
        
    //     int32_t numPairs = d_info[0];
    //     int32_t maxSeqLen = d_info[1]; 

    //     // Iterate over every pair of sequences
    //     // TODO: Parallelize - assign one block per alignment pair
    //     for (int pair = 0; pair < numPairs; ++pair) {
            
    //         // --- Initialization per Pair ---
    //         // HINT:
    //         // Think carefully,
    //         // 1. Should these variables be stored in registers or shared memory?
    //         //    (feel free to modify the baseline code)
    //         // 2. How many threads are needed to initialize shared memory efficiently?
    //         // 3. Should __syncthreads() be added after initialization?
    //         lastTile = false;
    //         maxScore = 0;
    //         int32_t currentPairPathLen = 0; 
    //         int32_t reference_idx = 0; 
    //         int32_t query_idx = 0;  

    //         // Calculate memory offsets for this pair
    //         int32_t refStart = (pair * 2) * maxSeqLen;
    //         int32_t qryStart = (pair * 2 + 1) * maxSeqLen;
    //         int32_t tbGlobalOffset = pair * (maxSeqLen * 2); 
            
    //         int32_t refTotalLen = d_seqLen[2 * pair];
    //         int32_t qryTotalLen = d_seqLen[2 * pair + 1];
               
            
    //         // -------------------------------------------------------
    //         // TILE LOOP: Align the sequence tile by tile
    //         // -------------------------------------------------------
    //         while (!lastTile) {
    //             // Determine Tile Size (Clip to sequence end)
    //             int32_t refLen = min(T, refTotalLen - reference_idx); 
    //             int32_t qryLen = min(T, qryTotalLen - query_idx); 
                
    //             // Check termination conditions
    //             // HINT:
    //             // 1. How many threads are needed here?
    //             if ((reference_idx + refLen == refTotalLen) && (query_idx + qryLen == qryTotalLen)) lastTile = true;

    //             // Reset Wavefront Scores (approx -infinity)
    //             // HINT: 
    //             // 1. How many threads are needed to efficiently initialize wf_scores?
    //             // 2. Is the initialization really necessary, or can it be omitted?
    //             for (int s = 0; s < 3 * (T + 1); ++s) wf_scores[s] = -9999;
                
    //             // Reset Max Score Tracking (for finding optimal overlap exit)
    //             best_ti = refLen; 
    //             best_tj = qryLen;

    //             // TODO: Load the reference and query segments from global memory into shared memory
    //             // HINT: Using memory coalescing
    //             // for (int s = ??; s < ??; s += ??) shared_ref[s] = d_seqs[??];
    //             // for (int s = ??; s < ??; s += ??) shared_qry[s] = d_seqs[??];

    //             // ---------------------------------------------------
    //             // WAVEFRONT SCORING LOOP (Diagonal Traversal)
    //             // ---------------------------------------------------
    //             for (int k = 0; k <= refLen + qryLen; ++k) {
                    
    //                 // Cyclic buffers for 3-wavefront dependency
    //                 int curr_k   = (k % 3) * (T + 1);
    //                 int pre_k    = ((k + 2) % 3) * (T + 1);
    //                 int prepre_k = ((k + 1) % 3) * (T + 1);

    //                 // Compute loop bounds for this diagonal
    //                 int i_start = max(0, k - qryLen);
    //                 int i_end   = min(refLen, k);

    //                 // TODO: Implement wavefront parallelism
    //                 // HINT: 
    //                 // 1. Assign each thread to a cell on the wavefront
    //                 // 2. What is the maximum possible length of wavefront? 
    //                 //    Can we always set blockSize to avoid situations 
    //                 //    where the wavefront size exceeds the number of available block threads?
    //                 for (int i = i_start; i <= i_end; ++i) {
    //                     int j = k - i; 
                        
    //                     int16_t score = -9999;
    //                     uint8_t direction = DIR_DIAG;

    //                     // -- Boundary Conditions --
    //                     if (i == 0 && j == 0) {
    //                         score = maxScore; // Score from the previous tile
    //                         maxScore = -9999; // reset max score
    //                     } 
    //                     else if (i == 0) {
    //                         score = wf_scores[pre_k + i] + GAP; // Gap from Left
    //                         direction = DIR_LEFT;
    //                     } 
    //                     else if (j == 0) {
    //                         score = wf_scores[pre_k + (i - 1)] + GAP; // Gap from Up
    //                         direction = DIR_UP;
    //                     } 
    //                     else {
    //                         // -- Inner Matrix Calculation --
    //                         // TODO: Replace this global memory access with shared memory
    //                         char r_char = d_seqs[refStart + reference_idx + (i - 1)];
    //                         char q_char = d_seqs[qryStart + query_idx + (j - 1)];
                            
    //                         int16_t score_diag = wf_scores[prepre_k + (i - 1)] + (r_char == q_char ? MATCH : MISMATCH);
    //                         int16_t score_up   = wf_scores[pre_k + (i - 1)] + GAP;
    //                         int16_t score_left = wf_scores[pre_k + i] + GAP;
                        
    //                         // Find Max (BONUS: Replace with DPX instructions)
    //                         score = score_diag;
    //                         direction = DIR_DIAG;
                            
    //                         if (score_up > score) { 
    //                             score = score_up; 
    //                             direction = DIR_UP; 
    //                         }
    //                         if (score_left > score) { 
    //                             score = score_left; 
    //                             direction = DIR_LEFT; 
    //                         }
    //                     }

    //                     // Write Score
    //                     wf_scores[curr_k + i] = score;
                        
    //                     // Write Direction (Only for inner cells, shifted index)
    //                     if (i > 0 && j > 0) {
    //                          tbDir[(i - 1) * T + (j - 1)] = direction;
    //                     }

    //                     // -- GACT Overlap Logic --
    //                     // Track the highest score in the overlap region (past T-O)
    //                     // TODO: Compute the max score and its indices for cells in the overlap region
    //                     // Sequentially, max score can be updated during iteration.
    //                     // For parallel execution, think about how to handle this safely.
    //                     // HINT: Parallel Reduction
    //                     if (!lastTile) {
    //                         if (i > (refLen - O) && j > (qryLen - O)) {
    //                            if (score >= maxScore) {
    //                                maxScore = score;
    //                                best_ti = i;
    //                                best_tj = j;
    //                            }
    //                         }
    //                     }                    
    //                 }
    //             } // End Wavefront Loop

    //             // ---------------------------------------------------
    //             // TRACEBACK & REVERSAL
    //             // ---------------------------------------------------
    //             // HINT:
    //             // 1. How many threads should be used for the traceback?
    //             // 2. Consider carefully whether the variable should be stored in registers or shared memory.

    //             // Determine where to start traceback (Overlap heuristic vs End of Tile)
    //             int ti = (!lastTile) ? best_ti : refLen;
    //             int tj = (!lastTile) ? best_tj : qryLen;

    //             // Determine how much we advanced in this tile
    //             int next_ref_advance = ti;
    //             int next_qry_advance = tj;

    //             // Traceback Backwards (End -> Start) into Shared Memory
    //             int localLen = 0;
                
    //             while (ti > 0 || tj > 0) {
    //                 uint8_t dir;

    //                 // Implicit boundary handling for top/left edges
    //                 if (ti == 0) { 
    //                     dir = DIR_LEFT; 
    //                 } else if (tj == 0) { 
    //                     dir = DIR_UP;   
    //                 } else {
    //                     // Fetch direction from DP table
    //                     dir = tbDir[(ti - 1) * T + (tj - 1)];
    //                 }

    //                 // Store to local temporary buffer
    //                 localPath[localLen] = dir;
    //                 localLen++;
                    
    //                 // Move coordinates
    //                 if (dir == DIR_DIAG) { ti--; tj--; } 
    //                 else if (dir == DIR_UP) { ti--; } 
    //                 else { tj--; }
    //             }

    //             // Write Forward (Start -> End) to Global Memory
    //             // Reverses the local path so the CPU gets it in forward order
    //             // TODO: Use memory coalescing to efficiently write the data to global memory
    //             for (int k = localLen - 1; k >= 0; --k) {
    //                 d_tb[tbGlobalOffset + currentPairPathLen] = localPath[k];
    //                 currentPairPathLen++;
    //             }
                
    //             // ---------------------------------------------------
    //             // ADVANCE TO NEXT TILE
    //             // ---------------------------------------------------
    //             reference_idx += next_ref_advance;
    //             query_idx     += next_qry_advance;

    //         } // End Tile Loop
    //         // HINT: Is __syncthreads() needed after each tile?
    //     } // End Pair Loop
    //     // HINT: Is __syncthreads() needed after each alignment?
    // }

    // -----------------------------------------------------------
    // MY IMPLEMENTATION
    // -----------------------------------------------------------

    // Use shared memory to store the reference and query segments involved in the tile
    __shared__ char shared_ref[T];
    __shared__ char shared_qry[T];

    // Tile sizing, cursor and other variables moved to shared mem
    __shared__ int32_t shared_refCursor;  
    __shared__ int32_t shared_qryCursor;  
    __shared__ int32_t shared_refLen;     
    __shared__ int32_t shared_qryLen;     
    __shared__ int32_t shared_pathLen;    
    __shared__ int32_t shared_localLen;   

    __shared__ int16_t  wf_cand_score[T + 1];
    __shared__ int32_t  wf_cand_ti[T + 1];
    __shared__ int32_t  wf_cand_tj[T + 1];

    int32_t numPairs = d_info[0]; // Not sure if this is needed later on?
    int32_t maxSeqLen = d_info[1]; 

    int pair = bx; // One block per pair
    if (pair >= numPairs) return; // Guard for now, maybe remove later if not needed

    // Calculate memory offsets for this pair
    int32_t refStart = (pair * 2) * maxSeqLen;
    int32_t qryStart = (pair * 2 + 1) * maxSeqLen;
    int32_t tbGlobalOffset = pair * (maxSeqLen * 2); 
    
    int32_t refTotalLen = d_seqLen[2 * pair];
    int32_t qryTotalLen = d_seqLen[2 * pair + 1];

    // --- Initialization per Pair ---
    // HINT:
    // Think carefully,
    // 1. Should these variables be stored in registers or shared memory?
    //    (feel free to modify the baseline code)
    // 2. How many threads are needed to initialize shared memory efficiently?
    // 3. Should __syncthreads() be added after initialization?

    // Thread 0 iniitializes shared mem variables
    if (tx == 0) {
        lastTile = false;
        maxScore = 0;
        shared_refCursor = 0;
        shared_qryCursor = 0;
        shared_pathLen = 0;
    }
    __syncthreads();
        
    // -------------------------------------------------------
    // TILE LOOP: Align the sequence tile by tile
    // -------------------------------------------------------
    while (!lastTile) {

        // --- Thread 0: compute tile dimensions and check termination ---
        if (tx == 0) {
            // Determine Shared mem Tile Size (Clip to sequence end)
            shared_refLen = min(T, refTotalLen - shared_refCursor);
            shared_qryLen = min(T, qryTotalLen - shared_qryCursor);
            // Check termination conditions
            // HINT: How many threads are needed here?
            if ((shared_refCursor + shared_refLen == refTotalLen) && (shared_qryCursor + shared_qryLen == qryTotalLen)){
                lastTile = true;
            } 
            // Reset Max Score Tracking (for finding optimal overlap exit)
            best_ti = shared_refLen;
            best_tj = shared_qryLen;
        }
        __syncthreads(); 

        // Reset Wavefront Scores (approx -infinity)
        // HINT: 
        // 1. How many threads are needed to efficiently initialize wf_scores?
        // 2. Is the initialization really necessary, or can it be omitted?
        for (int s = tx; s < 3 * (T + 1); s += blockDim.x) wf_scores[s] = -9999;
        
        // Load the reference and query segments from global memory into shared memory
        // HINT: Using memory coalescing
        for (int s = tx; s < shared_refLen; s += blockDim.x) shared_ref[s] = d_seqs[refStart + shared_refCursor + s];
        for (int s = tx; s < shared_qryLen; s += blockDim.x) shared_qry[s] = d_seqs[qryStart + shared_qryCursor + s];
        __syncthreads(); 

        // ---------------------------------------------------
        // WAVEFRONT SCORING LOOP (Diagonal Traversal)
        // ---------------------------------------------------
        for (int k = 0; k <= shared_refLen + shared_qryLen; ++k) {
            // Cyclic buffers for 3-wavefront dependency
            int curr_k   = (k % 3) * (T + 1);
            int pre_k    = ((k + 2) % 3) * (T + 1);
            int prepre_k = ((k + 1) % 3) * (T + 1);

            int i_start = max(0, k - shared_qryLen);
            int i_end = min(shared_refLen, k);
            int wf_len = i_end - i_start + 1;  // active cells on this diagonal

            // Each thread maps to one cell on the diagonal
            int i = i_start + tx;

            // Reset thread overlap candidate
            wf_cand_score[tx] = -9999;
            wf_cand_ti[tx] = -1; 
            wf_cand_tj[tx] = -1;

            if (i <= i_end) {
                int j = k - i;

                int16_t score = -9999;
                uint8_t direction = DIR_DIAG;

                // -- Boundary Conditions --
                if (i == 0 && j == 0) {
                    score = maxScore; // Score from the previous tile
                    maxScore = -9999; // reset max score
                } 
                else if (i == 0) {
                    score = wf_scores[pre_k + i] + GAP; // Gap from Left
                    direction = DIR_LEFT;
                } 
                else if (j == 0) {
                    score = wf_scores[pre_k + (i - 1)] + GAP; // Gap from Up
                    direction = DIR_UP;
                } 
                else {
                    // -- Inner Matrix Calculation --
                    // (DONE) Replace this global memory access with shared memory
                    char r_char = shared_ref[i - 1];   
                    char q_char = shared_qry[j - 1];

                    int16_t score_diag = wf_scores[prepre_k + (i - 1)] + (r_char == q_char ? MATCH : MISMATCH);
                    int16_t score_up   = wf_scores[pre_k + (i - 1)] + GAP;
                    int16_t score_left = wf_scores[pre_k + i] + GAP;
                        
                    // Find Max (BONUS: Replace with DPX instructions)
                    score = score_diag;
                    direction = DIR_DIAG;
                    
                    if (score_up > score) { 
                        score = score_up; 
                        direction = DIR_UP; 
                    }
                    if (score_left > score) { 
                        score = score_left; 
                        direction = DIR_LEFT; 
                    }
                }

                // Write Score
                wf_scores[curr_k + i] = score;
                
                // Write Direction (Only for inner cells, shifted index)
                if (i > 0 && j > 0) {
                        tbDir[(i - 1) * T + (j - 1)] = direction;
                }

                // Update cell score if it falls inside the overlap region
                if (!lastTile && i > (shared_refLen - O) && j > (shared_qryLen - O)) {
                    wf_cand_score[tx] = score;
                    wf_cand_ti[tx]    = i;
                    wf_cand_tj[tx]    = j;
                }
            }
            __syncthreads();  // wf_scores[curr_k] and wf_cand_* fully written

            // -- GACT Overlap Logic --
            // Track the highest score in the overlap region (past T-O)
            // (DONE) Compute the max score and its indices for cells in the overlap region
            // Sequentially, max score can be updated during iteration.
            // For parallel execution, think about how to handle this safely.
            // HINT: Parallel Reduction
            if (tx == 0 && !lastTile) {
                for (int s = 0; s < wf_len; ++s) {
                    if (wf_cand_ti[s] != -1 && wf_cand_score[s] >= maxScore) {
                        maxScore = wf_cand_score[s];
                        best_ti  = wf_cand_ti[s];
                        best_tj  = wf_cand_tj[s];
                    }
                }
            }
            __syncthreads();  // maxScore/best_ti/best_tj stable before next diagonal
        }
        // End wavefront loop

        // ---------------------------------------------------
        // TRACEBACK & REVERSAL
        // ---------------------------------------------------
        // HINT:
        // 1. How many threads should be used for the traceback?
        // 2. Consider carefully whether the variable should be stored in registers or shared memory.
        if (tx == 0) {
            // Determine where to start traceback (Overlap heuristic vs End of Tile)
            int ti = (!lastTile) ? best_ti : shared_refLen;
            int tj = (!lastTile) ? best_tj : shared_qryLen;

            // Determine how much we advanced in this tile
            int next_ref_advance = ti;
            int next_qry_advance = tj;

            // Traceback Backwards (End -> Start) into Shared Memory
            int localLen = 0;

            while (ti > 0 || tj > 0) {
                uint8_t dir;

                // Implicit boundary handling for top/left edges
                if (ti == 0) { 
                    dir = DIR_LEFT; 
                } else if (tj == 0) { 
                    dir = DIR_UP;   
                } else {
                    // Fetch direction from DP table
                    dir = tbDir[(ti - 1) * T + (tj - 1)];
                }

                // Store to local temporary buffer
                localPath[localLen] = dir;
                localLen++;
                
                // Move coordinates
                if (dir == DIR_DIAG) { ti--; tj--; } 
                else if (dir == DIR_UP) { ti--; } 
                else { tj--; }
            }

            shared_localLen   = localLen;
            shared_refCursor += next_ref_advance;
            shared_qryCursor += next_qry_advance;
        }
        __syncthreads(); 

        // Write Forward (Start -> End) to Global Memory
        // Reverses the local path so the CPU gets it in forward order
        // (DONE) Use memory coalescing to efficiently write the data to global memory
        for (int w = tx; w < shared_localLen; w += blockDim.x) {
            d_tb[tbGlobalOffset + shared_pathLen + w] = localPath[shared_localLen - 1 - w]; 
        }
        __syncthreads();

        if (tx == 0) {
            shared_pathLen += shared_localLen;
        } 
        __syncthreads();

    } // End Tile Loop
}

/** * Reconstructs the actual string alignment from the traceback paths (CIGAR-like data).
 * Converts directional codes (DIAG, UP, LEFT) into aligned strings with gaps.
 */
void GpuAligner::getAlignedSequences (TB_PATH& tb_paths) {

    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;
    
    int tb_length = longestLen << 1;
    
    // (DONE) Apply parallelism to this for loop
    // HINT: Remember to add the header
    tbb::parallel_for(0, numPairs, [&](int pair) {
        int tb_start = tb_length * pair;
        
        int seqId0 = 2 * pair;
        int seqId1 = 2 * pair + 1;
        std::string seq0 = seqs[seqId0].seq;
        std::string seq1 = seqs[seqId1].seq;
        std::string aln0 = "";
        std::string aln1 = "";
        int seqPos0 = 0;
        int seqPos1 = 0;

        // Iterate through the recorded path directions
        for (int i = tb_start; i < tb_start+tb_length; ++i) {
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
        seqs[seqId0].aln = aln0;
        seqs[seqId1].aln = aln1;
    });
}

void GpuAligner::clearAndReset () {
    cudaFree(d_seqs);
    cudaFree(d_seqLen);
    cudaFree(d_tb);
    seqs.clear();
    longestLen = 0;
    numPairs = 0;
}

/**
 * Main orchestration method.
 * 1. Allocates GPU memory
 * 2. Transfers data
 * 3. Launches Kernel
 * 4. Retrieves results and reconstructs alignment strings
 */
void GpuAligner::alignment () {

    // (DONE) make sure to appropriately set the values below
    int numBlocks = numPairs;  // i.e. number of thread blocks on the GPU
    int blockSize = T + 1; // i.e. number of GPU threads per thread block

    // 1. Allocate memory on Device
    allocateMem();
    
    // 2. Transfer sequence to device
    transferSequence2Device();
    
    // 3. Perform the alignment on GPU
    alignmentOnGPU<<<numBlocks, blockSize>>>(d_info, d_seqLen, d_seqs, d_tb);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
    
    // 4. Transfer the traceback path from device
    TB_PATH tb_paths = transferTB2Host();
    cudaDeviceSynchronize();
    
    // 5. Get the aligned sequence with traceback paths
    getAlignedSequences(tb_paths);
    
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