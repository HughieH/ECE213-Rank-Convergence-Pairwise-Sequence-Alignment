// ============================================================
// Fixed full NW global alignment kernel (wavefront)
// Fixes highlighted with "FIX #" comments.
//   FIX #1: Use DP_STRIDE = (maxSeqLen + 1) everywhere for DP-indexed tables.
//           Allocate/index d_tbDir with (DP_STRIDE * DP_STRIDE).
//   FIX #2: Initialize wf_scores (all 3 buffers) to -INF before diagonal loop.
//           Also clear the "current diagonal" buffer each k (safe).
//   FIX #3: Write a terminator 0 after traceback path so CPU stops correctly.
//           Also guard against overflow of d_tb capacity.
//   FIX #4: Shared memory alignment: use unsigned char for smem and align pointer.
// ============================================================

__global__ void alignmentOnGPU_NW_wavefront(
    const int32_t* __restrict__ d_info,    // [0]: numPairs, [1]: maxSeqLen (= longestLen)
    const int32_t* __restrict__ d_seqLen,  // [2*pair]: M, [2*pair+1]: N
    const char*    __restrict__ d_seqs,    // flattened sequences (stride maxSeqLen)
    uint8_t*       __restrict__ d_tb,      // traceback output, stride tb_stride per pair
    uint8_t*       __restrict__ d_tbDir    // full direction table (global), stride DP_STRIDE
) {
    int pair = blockIdx.x;
    int tx   = threadIdx.x;

    const int16_t MATCH    = 2;
    const int16_t MISMATCH = -1;
    const int16_t GAP      = -2;

    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    const int32_t numPairs  = d_info[0];
    const int32_t maxSeqLen = d_info[1];

    if (pair >= numPairs) return;

    // FIX #1: DP stride must include +1 state (0..len)
    const int32_t DP_STRIDE = maxSeqLen + 1;

    // Offsets into flattened input sequences (still stride maxSeqLen, indices 0..len-1)
    const int32_t refStart = (pair * 2) * maxSeqLen;
    const int32_t qryStart = (pair * 2 + 1) * maxSeqLen;

    const int32_t M = d_seqLen[2 * pair];      // ref length
    const int32_t N = d_seqLen[2 * pair + 1];  // qry length

    // Safety: if any length exceeds maxSeqLen, kernel would OOB
    if (M > maxSeqLen || N > maxSeqLen) return;

    // FIX #1: direction table offset uses DP_STRIDE^2
    const int32_t tbDirOffset = pair * DP_STRIDE * DP_STRIDE;

    // Traceback buffer stride on global memory.
    // You should allocate host/device d_tb with at least (2*maxSeqLen + 2) bytes per pair
    // to fit terminator. (Or store len separately.)
    const int32_t tb_stride = (maxSeqLen * 2 + 2);
    const int32_t tbGlobalOffset = pair * tb_stride;

    // --------------------------
    // Shared memory layout
    // --------------------------
    // Need:
    //   wf_scores: 3*(DP_STRIDE) int16
    //   shared_ref: maxSeqLen chars
    //   shared_qry: maxSeqLen chars
    //
    // FIX #4: alignment-safe shared memory pointer arithmetic
    extern __shared__ unsigned char smem_u8[];
    // wf_scores starts at smem_u8 (CUDA shared memory is at least 16B aligned, but we keep it clean)
    int16_t* wf_scores = reinterpret_cast<int16_t*>(smem_u8);
    char* shared_ref   = reinterpret_cast<char*>(wf_scores + 3 * DP_STRIDE);
    char* shared_qry   = shared_ref + maxSeqLen;

    // Load sequences into shared memory
    for (int i = tx; i < M; i += blockDim.x) shared_ref[i] = d_seqs[refStart + i];
    for (int j = tx; j < N; j += blockDim.x) shared_qry[j] = d_seqs[qryStart + j];
    __syncthreads();

    // FIX #2: Initialize all wf_scores buffers to -INF
    // Use a value small enough so adding MATCH/MISMATCH/GAP doesn't overflow
    const int16_t NEG_INF = (int16_t)-30000;
    for (int idx = tx; idx < 3 * DP_STRIDE; idx += blockDim.x) {
        wf_scores[idx] = NEG_INF;
    }
    __syncthreads();

    // Main wavefront over diagonals k = i + j
    for (int k = 0; k <= M + N; ++k) {

        const int curr_k   = (k % 3) * DP_STRIDE;
        const int pre_k    = ((k + 2) % 3) * DP_STRIDE;  // k-1
        const int prepre_k = ((k + 1) % 3) * DP_STRIDE;  // k-2

        // FIX #2: Clear current diagonal buffer (safe; avoids stale reads in weird edge ranges)
        for (int idx = tx; idx < DP_STRIDE; idx += blockDim.x) {
            wf_scores[curr_k + idx] = NEG_INF;
        }
        __syncthreads();

        const int i_start = max(0, k - N);
        const int i_end   = min(M, k);

        for (int i = i_start + tx; i <= i_end; i += blockDim.x) {
            const int j = k - i;

            int16_t score = NEG_INF;
            uint8_t direction = DIR_DIAG;

            if (i == 0 && j == 0) {
                score = 0;
                // no direction stored for (0,0)
            } else if (i == 0) {
                // top row: only LEFT moves (gap in X / consume Y)
                score = (int16_t)(j * GAP);
                direction = DIR_LEFT;
            } else if (j == 0) {
                // left col: only UP moves (consume X / gap in Y)
                score = (int16_t)(i * GAP);
                direction = DIR_UP;
            } else {
                // Inner recurrence
                const char r_char = shared_ref[i - 1];
                const char q_char = shared_qry[j - 1];

                const int16_t s_diag = wf_scores[prepre_k + (i - 1)] + (r_char == q_char ? MATCH : MISMATCH);
                const int16_t s_up   = wf_scores[pre_k    + (i - 1)] + GAP;
                const int16_t s_left = wf_scores[pre_k    + i]       + GAP;

                // Deterministic tie-break: DIAG > UP > LEFT (example)
                score = s_diag;
                direction = DIR_DIAG;
                if (s_up > score) { score = s_up; direction = DIR_UP; }
                if (s_left > score) { score = s_left; direction = DIR_LEFT; }

                // If you want to use your max_score() helper, you can replace above with:
                // max_score(s_diag, s_up, s_left, score, direction, DIR_DIAG, DIR_UP, DIR_LEFT);
            }

            // Write score for this diagonal
            wf_scores[curr_k + i] = score;

            // FIX #1: direction table uses DP_STRIDE indexing (0..M,0..N)
            if (i > 0 || j > 0) {
                d_tbDir[tbDirOffset + i * DP_STRIDE + j] = direction;
            }
        }

        __syncthreads(); // barrier between diagonals
    }

    // --------------------------
    // Traceback: from (M,N) -> (0,0)
    // --------------------------
    if (tx == 0) {
        int ti = M, tj = N;
        int localLen = 0;

        // FIX #3: ensure we don't overflow d_tb.
        // Max steps is (M+N), plus terminator.
        const int max_write = tb_stride - 1; // leave space for terminator

        while ((ti > 0 || tj > 0) && localLen < max_write) {
            uint8_t dir;
            if (ti == 0) {
                dir = DIR_LEFT;
            } else if (tj == 0) {
                dir = DIR_UP;
            } else {
                dir = d_tbDir[tbDirOffset + ti * DP_STRIDE + tj];
                // Safety: if dir got corrupted/uninitialized, break to avoid infinite loop
                if (dir != DIR_DIAG && dir != DIR_UP && dir != DIR_LEFT) break;
            }

            // write path in reverse order first
            d_tb[tbGlobalOffset + localLen] = dir;
            localLen++;

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

        // FIX #3: write terminator so CPU stops
        d_tb[tbGlobalOffset + localLen] = 0;
    }
}