#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <omp.h> // 2d


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
 * TODO: Optimize using shared memory, wavefront parallelism, and memory coalescing.
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
    
    // GACT Parameters
    // TODO DONE: Adjust the tile size and overlap region to explore the tradeoff between speed and accuracy
    const int T = 160;        // Tile size (originally 10)
    const int O = 50;         // Overlap between tiles (originally 3) 
    
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
    __shared__ int32_t maxScore;          // use int for atomicMax
    __shared__ int16_t tileOriginScore;   // score to seed (0,0) from previous tile
    __shared__ int32_t best_ti;
    __shared__ int32_t best_tj;


    // HINT: Use shared memory to store the reference and query segments involved in the tile
    // (2b) Cache the tile's ref/qry characters in shared memory.
    __shared__ char shared_ref[T];
    __shared__ char shared_qry[T];

    // (2b) Per-pair state shared across threads (tx==0 computes DP; other threads help with coalesced loads/stores)
    __shared__ int32_t currentPairPathLen;
    __shared__ int32_t reference_idx;
    __shared__ int32_t query_idx;
    __shared__ int32_t refStart;
    __shared__ int32_t qryStart;
    __shared__ int32_t tbGlobalOffset;
    __shared__ int32_t refTotalLen;
    __shared__ int32_t qryTotalLen;
    __shared__ int32_t next_ref_advance_sh;
    __shared__ int32_t next_qry_advance_sh;
    __shared__ int32_t localLenSh;
    __shared__ int32_t basePathLenSh;

    // Read meta info for this kernel launch
    int32_t numPairs = d_info[0];
    int32_t maxSeqLen = d_info[1];

    // Iterate over every pair of sequences
    // TODO DONE: Parallelize – assign one block per alignment pair
    for (int pair = bx; pair < numPairs; pair += gridDim.x) {

        // --- Initialization per Pair ---
        if (tx == 0) {
            lastTile = false;
            maxScore = 0;

            currentPairPathLen = 0;
            reference_idx = 0;
            query_idx = 0;

            // Calculate memory offsets for this pair
            refStart = (pair * 2) * maxSeqLen;
            qryStart = (pair * 2 + 1) * maxSeqLen;
            tbGlobalOffset = pair * (maxSeqLen * 2);

            refTotalLen = d_seqLen[2 * pair];
            qryTotalLen = d_seqLen[2 * pair + 1];
        }
        __syncthreads();

    // -------------------------------------------------------
    // TILE LOOP: Align the sequence tile by tile
    // -------------------------------------------------------
    while (!lastTile) {
        // Determine Tile Size (Clip to sequence end)
        int32_t refLen = min(T, refTotalLen - reference_idx);
        int32_t qryLen = min(T, qryTotalLen - query_idx);

        // Check termination conditions
        // HINT:
        // 1. How many threads are needed here?
        if (tx == 0) {
            if ((reference_idx + refLen == refTotalLen) && (query_idx + qryLen == qryTotalLen)) lastTile = true;

            // Reset Max Score Tracking (for finding optimal overlap exit)
            best_ti = refLen;
            best_tj = qryLen;
        }

        // Reset Wavefront Scores (approx -infinity)
        // HINT:
        // 1. How many threads are needed to efficiently initialize wf_scores?
        // 2. Is the initialization really necessary, or can it be omitted?
        for (int s = tx; s < 3 * (T + 1); s += blockDim.x) wf_scores[s] = -9999;

        // TODO DONE: Load the reference and query segments from global memory into shared memory
        // HINT: Using memory coalescing
        for (int s = tx; s < refLen; s += blockDim.x) shared_ref[s] = d_seqs[refStart + reference_idx + s];
        for (int s = tx; s < qryLen; s += blockDim.x) shared_qry[s] = d_seqs[qryStart + query_idx + s];

        __syncthreads();

        int32_t next_ref_advance = 0;
        int32_t next_qry_advance = 0;

        if (tx == 0) {
            // Save the score from the previous tile for (0,0)
            tileOriginScore = (int16_t)maxScore;
            // Now repurpose maxScore for overlap max tracking in THIS tile
            maxScore = -9999;
        }
        __syncthreads();

        // ---------------------------------------------------
        // WAVEFRONT SCORING LOOP (Diagonal Traversal)  [2c]
        // ---------------------------------------------------
        for (int k = 0; k <= refLen + qryLen; ++k) {

            int curr_k   = (k % 3) * (T + 1);
            int pre_k    = ((k + 2) % 3) * (T + 1);
            int prepre_k = ((k + 1) % 3) * (T + 1);

            int i_start = max(0, k - qryLen);
            int i_end   = min(refLen, k);

            // Each thread computes multiple cells on this diagonal
            for (int i = i_start + tx; i <= i_end; i += blockDim.x) {
                int j = k - i;

                int16_t score = -9999;
                uint8_t direction = DIR_DIAG;

                if (i == 0 && j == 0) {
                    score = tileOriginScore;   // seed from previous tile
                }
                else if (i == 0) {
                    score = wf_scores[pre_k + i] + GAP;
                    direction = DIR_LEFT;
                }
                else if (j == 0) {
                    score = wf_scores[pre_k + (i - 1)] + GAP;
                    direction = DIR_UP;
                }
                else {
                    // Use shared memory (2b)
                    char r_char = shared_ref[i - 1];
                    char q_char = shared_qry[j - 1];

                    int16_t score_diag = wf_scores[prepre_k + (i - 1)] + (r_char == q_char ? MATCH : MISMATCH);
                    int16_t score_up   = wf_scores[pre_k + (i - 1)] + GAP;
                    int16_t score_left = wf_scores[pre_k + i] + GAP;

                    score = score_diag;
                    direction = DIR_DIAG;

                    if (score_up > score)   { score = score_up;   direction = DIR_UP; }
                    if (score_left > score) { score = score_left; direction = DIR_LEFT; }
                }

                wf_scores[curr_k + i] = score;

                if (i > 0 && j > 0) {
                    tbDir[(i - 1) * T + (j - 1)] = direction;
                }

                // Overlap max tracking (atomic)
                if (!lastTile) {
                    if (i > (refLen - O) && j > (qryLen - O)) {
                        int old = atomicMax(&maxScore, (int)score);
                        if ((int)score > old) {
                            // "Winner" writes indices (benign races only when very close in time)
                            best_ti = i;
                            best_tj = j;
                        }
                    }
                }
            }

            // Critical: next diagonal reads results produced by this diagonal
            __syncthreads();
        }
        // End Wavefront Loop
        if (tx == 0) {
                        // ---------------------------------------------------
                        // TRACEBACK & REVERSAL
                        // ---------------------------------------------------
                        // HINT:
                        // 1. How many threads should be used for the traceback?
                        // 2. Consider carefully whether the variable should be stored in registers or shared memory.

                        // Determine where to start traceback (Overlap heuristic vs End of Tile)
                        int ti = (!lastTile) ? best_ti : refLen;
                        int tj = (!lastTile) ? best_tj : qryLen;

                        // Determine how much we advanced in this tile
                        next_ref_advance = ti;
                        next_qry_advance = tj;

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

                
            // Expose results to the whole block for coalesced stores / next-tile loads
            basePathLenSh = currentPairPathLen;
            localLenSh = localLen;
            next_ref_advance_sh = next_ref_advance;
            next_qry_advance_sh = next_qry_advance;
        }
        __syncthreads();

        // Write Forward (Start -> End) to Global Memory
        // Reverses the local path so the CPU gets it in forward order
        // TODO DONE: Use memory coalescing to efficiently write the data to global memory
        for (int k = tx; k < localLenSh; k += blockDim.x) {
            d_tb[tbGlobalOffset + basePathLenSh + k] = localPath[localLenSh - 1 - k];
        }
        __syncthreads();

        // ---------------------------------------------------
        // ADVANCE TO NEXT TILE
        // ---------------------------------------------------
        if (tx == 0) {
            currentPairPathLen += localLenSh;
            reference_idx += next_ref_advance_sh;
            query_idx     += next_qry_advance_sh;
        }
        __syncthreads();

    } // End Tile Loop
    // HINT: Is __syncthreads() needed after each tile?

    __syncthreads();
    
    }

    // HINT: Is __syncthreads() needed after each alignment?
}

/** * Reconstructs the actual string alignment from the traceback paths (CIGAR-like data).
 * Converts directional codes (DIAG, UP, LEFT) into aligned strings with gaps.
 */
void GpuAligner::getAlignedSequences (TB_PATH& tb_paths) {

    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;
    
    int tb_length = longestLen << 1;
    
    // TODO DONE: Apply parallelism to this for loop
    // HINT: Remember to add the header
    #pragma omp parallel for schedule(static)
    for (int pair = 0; pair < numPairs; ++pair) {
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
    }
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

    // TODO DONE: make sure to appropriately set the values below
    int numBlocks = numPairs;  // i.e. number of thread blocks on the GPU
    int blockSize = 256; // i.e. number of GPU threads per thread block

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