#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

// Configuration for scoring
const int MATCH_SCORE = 2;
const int MISMATCH_SCORE = -1;
const int GAP_SCORE = -2;

static int8_t aa_to_idx_host[256];
static bool aa_inited = false;

static void init_aa_tables() {
    if (aa_inited) return;
    aa_inited = true;

    for (int i = 0; i < 256; i++) aa_to_idx_host[i] = -1;
    const char* AA = "ARNDCQEGHILKMFPSTWYV";
    for (int i = 0; i < 20; i++) aa_to_idx_host[(unsigned char)AA[i]] = (int8_t)i;
}

static const int8_t blosum62_host[20 * 20] = {
    // same 400 numbers as in alignment.cu
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

struct AlignmentPair {
    string seqA;
    string seqB;
    string name; // Optional, for reporting
};

// Strip columns where BOTH sequences have '-' (MSA-only artifact columns)
pair<string, string> stripGapOnlyColumns(const string& a, const string& b) {
    string outA, outB;
    size_t len = min(a.size(), b.size());
    for (size_t i = 0; i < len; ++i) {
        if (a[i] == '-' && b[i] == '-') continue;
        outA += a[i];
        outB += b[i];
    }
    return {outA, outB};
}

// Function to calculate alignment score (+ check scoring mode)
long long calculateScore(const string& a, const string& b, bool isProtein) {
    if (a.length() != b.length()) {
        fprintf(stderr, "WARNING: aligned strings differ in length (%zu vs %zu) — alignment may be malformed\n",
                a.length(), b.length());
    }
    long long score = 0;
    size_t len = min(a.length(), b.length());

    for (size_t i = 0; i < len; ++i) {
        char c1 = a[i];
        char c2 = b[i];

        // Gap in either sequence
        if (c1 == '-' || c2 == '-') {
            score += GAP_SCORE;
            continue;
        }

        // 'N' in either position, set score to 0 (ambiguous nucleotide)
        if (!isProtein && (c1 == 'N' || c2 == 'N')) {
            score += 0;
            continue;
        }

        if (!isProtein) {
            score += (c1 == c2) ? MATCH_SCORE : MISMATCH_SCORE;
        } else {
            int8_t i1 = aa_to_idx_host[(unsigned char)c1];
            int8_t i2 = aa_to_idx_host[(unsigned char)c2];
            score += (i1 >= 0 && i2 >= 0) ? blosum62_host[i1 * 20 + i2] : MISMATCH_SCORE;
        }
    }
    return score;
}

// Helper to read FASTA or line-based files into pairs
vector<AlignmentPair> readAlignments(const string& filepath) {
    vector<AlignmentPair> alignments;
    ifstream file(filepath);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filepath << endl;
        exit(1);
    }

    // Temporary struct to hold parsed FASTA records
    struct FastaRecord {
        string header;
        string sequence;
    };

    vector<FastaRecord> records;
    string line;
    string currentHeader;
    string currentSeq;
    bool parsingStarted = false;

    // 1. Read the entire file and parse into FastaRecords
    while (getline(file, line)) {
        // Remove carriage returns if on Windows/mixed files
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (line[0] == '>') {
            // Save previous record if it exists
            if (parsingStarted) {
                records.push_back({currentHeader, currentSeq});
                currentSeq = ""; // Reset for next
            }
            // Start new record
            currentHeader = line.substr(1); // Remove '>'
            parsingStarted = true;
        } else {
            // It's a sequence line (append to handle multiline FASTA)
            currentSeq += line;
        }
    }
    // Push the very last record found
    if (parsingStarted) {
        records.push_back({currentHeader, currentSeq});
    }

    // 2. Group records into pairs (1&2, 3&4, etc.)
    for (size_t i = 0; i < records.size(); i += 2) {
        if (i + 1 < records.size()) {
            // We have a pair (i and i+1)
            alignments.push_back({
                records[i].sequence,     // Seq 1
                records[i+1].sequence,   // Seq 2
                records[i].header        // Use the first header as the Pair Name
            });
        } else {
            cerr << "Warning: File " << filepath << " has an odd number of sequences. "
                 << "Ignoring the last orphan sequence: " << records[i].header << endl;
        }
    }

    return alignments;
}

int main(int argc, char* argv[]) {
    string refFile, estFile;

    // 1. Setup Program Options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("reference,r", po::value<string>(&refFile)->required(), "Reference alignment file")
        ("estimate,e",  po::value<string>(&estFile)->required(), "Estimate alignment file (To Evaluate)")
        ("protein,p",   po::bool_switch(),                       "Force protein scoring (BLOSUM62)")
        ("verbose,v",                                            "Show all pair-wise comparisons")
        ("help,h",                                               "Produce help message");

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& ex) {
        if (argc == 1) {
            std::cerr << desc << std::endl;
            return 0;
        }
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }

    bool verbose   = vm.count("verbose");
    bool isProtein = vm["protein"].as<bool>();

    // Initialize BLOSUM62 lookup tables if protein mode
    if (isProtein) {
        init_aa_tables();
        cout << "Protein scoring mode enabled, using BLOSUM62 matrix" 
         << "\n" << endl;
    }
    
    // 2. Read Files
    if (verbose) cout << "Reading Estimate file: " << estFile << "..." << endl;
    vector<AlignmentPair> estAligns = readAlignments(estFile);

    if (verbose) cout << "Reading Reference file: " << refFile << "..." << endl;
    vector<AlignmentPair> refAligns = readAlignments(refFile);

    // 3. Process Comparisons
    size_t count = min(estAligns.size(), refAligns.size());
    if (count == 0) {
        cerr << "Error: One of the files contains no valid alignment pairs." << endl;
        return 1;
    }

    double totalPercentageLoss = 0.0;
    int validComparisons = 0;

    if (verbose) {
        cout << "\n" << string(70, '-') << endl;
        cout << left << setw(10) << "ID"
             << setw(15) << "Ref Score"
             << setw(15) << "Est Score"
             << setw(15) << "Diff"
             << setw(15) << "% Loss" << endl;
        cout << string(70, '-') << endl;
    }

    for (size_t i = 0; i < count; ++i) {
        AlignmentPair& est = estAligns[i];
        AlignmentPair& ref = refAligns[i];

        // Strip gap-only columns (MSA artifact columns) before scoring
        auto [refA, refB] = stripGapOnlyColumns(ref.seqA, ref.seqB);
        auto [estA, estB] = stripGapOnlyColumns(est.seqA, est.seqB);

        // Two valid alignments of the same sequences may have different total
        // lengths (different gap placements), so truncating by column count
        // would compare different portions of the sequences
        long long scoreEst = calculateScore(estA, estB, isProtein);  
        long long scoreRef = calculateScore(refA, refB, isProtein);  

        long long diff = scoreRef - scoreEst;

        // Calculate Percentage Difference
        // Formula: (Ref - Est) / |Ref| * 100
        double percentLoss = 0.0;

        if (scoreRef != 0) {
            percentLoss = (double)diff / (double)abs(scoreRef) * 100.0;
        } else if (diff != 0) {
            // Edge case: Reference score is 0 but estimate differs — log and skip
            fprintf(stderr, "WARNING: pair %zu has ref score 0 but diff=%lld — skipping percent loss\n",
                    i + 1, diff);
        }

        totalPercentageLoss += percentLoss;
        validComparisons++;

        if (verbose) {
            cout << left << setw(10) << i+1
                 << setw(15) << scoreRef
                 << setw(15) << scoreEst
                 << setw(15) << diff
                 << fixed << setprecision(2) << percentLoss << "%" << endl;
        }
    }

    if (verbose) cout << string(70, '-') << endl;

    // 4. Final Average
    if (validComparisons > 0) {
        double avgLoss = totalPercentageLoss / validComparisons;
        cout << "Valid Comparisons: " << validComparisons << " pairs. ";
        cout << "Average Percentage Score Loss: " << avgLoss << "%\n" << endl;
    } else {
        cout << "No valid comparisons made." << endl;
    }

    return 0;
}
