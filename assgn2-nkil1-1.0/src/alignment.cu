#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <tbb/parallel_for.h>

__device__ __constant__ int8_t d_aa_to_idx[256];
__device__ __constant__ int8_t d_blosum62[20*20];

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
    int tb_length = longestLen * 2 + 2; 
    err = cudaMalloc(&d_tb, numPairs * tb_length * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

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
    int tb_length = longestLen * 2 + 2;
    std::vector<uint8_t> h_tb (tb_length * numPairs, 0);
    
    err = cudaMemcpy(d_tb, h_tb.data(), tb_length * numPairs * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 5. Transfer Meta Info
    std::vector<int32_t> h_info(3);
    h_info[0] = numPairs;
    h_info[1] = longestLen;

    // if any char is not A/C/G/T treat as protein
    int isProtein = 0;
    for (const auto& s : seqs) {
        for (char c : s.seq) {
            if (c!='A' && c!='C' && c!='G' && c!='T' && c!='N') { 
                isProtein = 1;
                break;
            }
        }
        if (isProtein) break;
    }
    h_info[2] = isProtein;

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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#include <cuda/ptx>
#endif

__device__ __forceinline__ void max_score(
    int16_t score_diag, int16_t score_up, int16_t score_left, int16_t &score_out,
    uint8_t &dir_out, uint8_t DIR_DIAG, uint8_t DIR_UP, uint8_t DIR_LEFT
) {
// have to add __CUDA_ARCH__ >= 900 to avoind compilation error on A30
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)    
    uint32_t du = score_diag | (score_up << 16);

    // put UP into lane0 (also lane1, but lane1 doesn't matter)
    uint32_t c  = score_up | (score_up<< 16);

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

#else // smae as the original code
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
 * CUDA Kernel: Full NW global alignment on the GPU.
// 1. Removed GACT tile loop
// 2. Removed overlap/maxScore/best_ti/best_tj logic
// 3. tbDir now covers the full M*N table (not T*T tile)
//      Because M*N can be large, tbDir lives in global memory (d_tbDir),
// 4. Traceback walks from (M,N) to (0,0) over the full tbDir table
// 5. Fix: wf_scores moved to global memory (d_wf) because shared mem is not large enough for big seq
//    smem request was ~78KB for longestLen=9941, but A30 only has 48K per block
*/
__global__ void alignmentOnGPU (
    int32_t* d_info,       // [0]: numPairs, [1]: maxSeqLen
    int32_t* d_seqLen,     // Array of sequence lengths
    char* d_seqs,          // Flat array of sequences
    uint8_t* d_tb,         // Output traceback paths
    uint8_t* d_tbDir,      // full N*M direction table, global mem
    int16_t* d_wf
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

    int32_t numPairs = d_info[0];
    int32_t maxSeqLen = d_info[1];
    int32_t isProtein = d_info[2];

    int pair = bx;
    if (pair >= numPairs) return;

    // Calculate memory offsets for this pair
    int32_t refStart = (pair * 2) * maxSeqLen;
    int32_t qryStart = (pair * 2 + 1) * maxSeqLen;

    int32_t DP_STRIDE = maxSeqLen + 1;
    int32_t tbDirOffset = pair * DP_STRIDE * DP_STRIDE;  // full table

    int32_t tb_stride = maxSeqLen * 2 + 2;
    int32_t tbGlobalOffset = pair * tb_stride; 

    int32_t wfOffset       = pair * 3 * DP_STRIDE;

    int32_t M = d_seqLen[2 * pair];      // ref len
    int32_t N = d_seqLen[2 * pair + 1];  // qry len
    if (M > maxSeqLen || N > maxSeqLen) return;

    // ---------------------------------------------------
    // Shared memory
    // ---------------------------------------------------
    extern __shared__ char smem[];
    char* shared_ref = smem;
    char* shared_qry = shared_ref + maxSeqLen;

    // Load the reference and query segments from global memory into shared memory
    // HINT: Using memory coalescing
    for (int i = tx; i < M; i += blockDim.x) shared_ref[i] = d_seqs[refStart + i];
    for (int j = tx; j < N; j += blockDim.x) shared_qry[j] = d_seqs[qryStart + j];
    __syncthreads();

    // Initialize global wf buffer for this pair
    int16_t* wf_scores = d_wf + wfOffset;
    for (int idx = tx; idx < 3 * DP_STRIDE; idx += blockDim.x) {
        wf_scores[idx] = -9999;
    }
    __syncthreads();

    // -------------------------------------------------------
    // Single diagonal loop over all M+N diagonals
    // -------------------------------------------------------
    for (int k = 0; k <= M + N; ++k) {

        // Cyclic buffers for 3-wavefront dependency
        int curr_k   = (k % 3) * DP_STRIDE;
        int pre_k    = ((k + 2) % 3) * DP_STRIDE;
        int prepre_k = ((k + 1) % 3) * DP_STRIDE;

        // TODO: should we clear current diagonal buffer?
        for (int idx = tx; idx < DP_STRIDE; idx += blockDim.x) {
            wf_scores[curr_k + idx] = -9999;
        }
        __syncthreads();

        // Compute loop bounds for this diagonal
        int i_start = max(0, k - N);
        int i_end   = min(M, k);

        // Wavefront parallelism
        for (int i = i_start + tx; i <= i_end; i += blockDim.x) {
            int j = k - i;

            int16_t score = -9999;
            uint8_t direction = DIR_DIAG;

            // -- Boundary Conditions --
            if (i == 0 && j == 0) {
                score = 0;
                // no direction needed for origin
            }
            else if (i == 0) {
                // NW base case: top row = LEFT gaps
                score = j * GAP;
                direction = DIR_LEFT;
            }
            else if (j == 0) {
                // NW base case: left col = UP gaps
                score = i * GAP;
                direction = DIR_UP;
            }
            else {
                // -- Inner Matrix Calculation --
                char r_char = shared_ref[i - 1];
                char q_char = shared_qry[j - 1];
                            
                int16_t diag_base = wf_scores[prepre_k + (i - 1)];
                int16_t score_diag;

                if (!isProtein) {
                    score_diag = diag_base + (r_char == q_char ? MATCH : MISMATCH);
                } else {
                    int8_t ri = d_aa_to_idx[(unsigned char)r_char];
                    int8_t qi = d_aa_to_idx[(unsigned char)q_char];
                    // minimal handling for unknown letters: treat as mismatch
                    int16_t sub = (ri >= 0 && qi >= 0) ? (int16_t)d_blosum62[ri * 20 + qi] : (int16_t)MISMATCH;
                    score_diag = diag_base + sub;
                }

                int16_t score_up   = wf_scores[pre_k + (i - 1)] + GAP;
                int16_t score_left = wf_scores[pre_k + i] + GAP;

                max_score(score_diag, score_up, score_left, score, direction, DIR_DIAG, DIR_UP, DIR_LEFT);
            }

            // Write Score
            wf_scores[curr_k + i] = score;

            // Write Direction to global tbDir table (not shared T*T)
            if (i > 0 || j > 0) {   // skip (0,0)
                d_tbDir[tbDirOffset + i * DP_STRIDE + j] = direction;
            }
        }
        __syncthreads();  // barrier between diagonals
    } // End Wavefront Loop

    // ---------------------------------------------------
    // TRACEBACK & REVERSAL
    //  walks full (M,N)->(0,0) path in global tbDir
    // ---------------------------------------------------
    if (tx == 0) {
        int localLen = 0;

        int ti = M, tj = N;
        while ((ti > 0 || tj > 0) && localLen < tb_stride - 1) {
            uint8_t dir;
            // Implicit boundary handling for top/left edges
            if (ti == 0) { 
                dir = DIR_LEFT; 
            } else if (tj == 0) { 
                dir = DIR_UP;   
            } else {
                // Fetch direction from DP table
                dir = d_tbDir[tbDirOffset + ti * DP_STRIDE + tj];
            }

            // write in reverse order first
            d_tb[tbGlobalOffset + localLen] = dir;
            localLen++;

            // Move coordinates
            if (dir == DIR_DIAG) { ti--; tj--; } 
            else if (dir == DIR_UP) { ti--; } 
            else { tj--; }
        }

        // reverse in-place
        for (int lo = 0, hi = localLen - 1; lo < hi; lo++, hi--) {
            uint8_t tmp = d_tb[tbGlobalOffset + lo];
            d_tb[tbGlobalOffset + lo] = d_tb[tbGlobalOffset + hi];
            d_tb[tbGlobalOffset + hi] = tmp;
        }
        d_tb[tbGlobalOffset + localLen] = 0;
    }
    
}


// 1. Allocate d_tbDir (full direction table, global mem)
// 2. Pass dynamic shared memory size (wf_scores + ref + qry)
// 3. Pass d_tbDir to kernel
// 4. Free d_tbDir after use

void GpuAligner::alignment() {

    // TODO: make sure to appropriately set the values below
    int numBlocks = numPairs;  // i.e. number of thread blocks on the GPU
    int blockSize = 256; // i.e. number of GPU threads per thread block

    // 1. Allocate memory on Device
    allocateMem();
    
    // 2. Transfer sequence to device
    transferSequence2Device();

    initProteinTablesOnce();
    alignmentOnGPU<<<numBlocks, blockSize>>>(d_info, d_seqLen, d_seqs, d_tb);

    const int DP_STRIDE = (size_t)longestLen + 1;

    // 3. allocate tbDir locally
    uint8_t* d_tbDir_local = nullptr;
    size_t dir_bytes = (size_t)numPairs * DP_STRIDE * DP_STRIDE * sizeof(uint8_t);
    cudaError_t err = cudaMalloc(&d_tbDir_local, dir_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_tbDir): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemset(d_tbDir_local, 0, dir_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (memset d_tbDir): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // 4. allocate wf buffer locally
    int16_t* d_wf_local = nullptr;
    size_t wf_bytes = (size_t)numPairs * 3 * DP_STRIDE * sizeof(int16_t);
    err = cudaMalloc(&d_wf_local, wf_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR (d_wf): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    

    // dynamic shared memory = shared_ref + shared_qry
    size_t smem_bytes = 2 * longestLen * sizeof(char);
    printf("longestLen=%d smem_bytes=%zu\n", longestLen, smem_bytes);

    // 5. Perform the alignment on GPU
    alignmentOnGPU<<<numBlocks, blockSize, smem_bytes>>>(d_info, d_seqLen, d_seqs, d_tb, d_tbDir_local, d_wf_local);

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
    //     exit(1);
    // }
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
    
    // 6. Transfer the traceback path from device
    TB_PATH tb_paths = transferTB2Host();
    cudaDeviceSynchronize();
    
    // 7. Get the aligned sequence with traceback paths
    getAlignedSequences(tb_paths);

    // 8. Free
    cudaFree(d_tbDir_local);
    cudaFree(d_wf_local);

}


//  The following functions are the same as hw

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
        seqs[seqId0].aln = std::move(aln0);
        seqs[seqId1].aln = std::move(aln1);
    });
}

void GpuAligner::clearAndReset () {
    cudaFree(d_seqs);
    cudaFree(d_seqLen);
    cudaFree(d_tb);
    // cudaFree(d_tbDir);
    // cudaFree(d_wf);
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