#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

mkdir -p build
cd build
cmake ..
make -j4
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/tbb_cmake_build/tbb_cmake_build_subdir_release



# ## Basic run to align the first 10 pairs (20 sequences) in the sequences.fa in batches of 2 reads
# ## HINT: may need to change values for the assignment tasks. You can create a sequence of commands
# ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1500 -T 8 --output alignment.fa

# ## Debugging with Compute Sanitizer
# ## Run this command to detect illegal memory accesses (out-of-bounds reads/writes) and race conditions.
# # compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 1 --output alignment.fa

# ## For debugging and evaluate the alignment accuracy
# ./check_alignment --raw ../data/sequences.fa --alignment alignment.fa                  # add -v to see failed result instead of a summary
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment.fa # add -v to see pair-wise comparisons


# echo "-------- No Compute Sanitizer run --------"
# ./aligner --sequence ../data/sequences.fa --maxPairs 200 --batchSize 50 -T 8 --output alignment.fa
# ./check_alignment --raw ../data/sequences.fa --alignment alignment.fa
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment.fa

# -- sars_20000 dataset check --
./aligner --sequence ../data/sars_20000.fa --maxPairs 200 --batchSize 150 -T 8 --output ../data/sars_alignment.fa
./check_alignment --raw ../data/sars_20000.fa --alignment ../data/sars_alignment.fa
./compare_alignment --reference ../data/sars_20000.msa --estimate ../data/sars_alignment.fa

# echo "-------- Compute Sanitizer run ------"
# compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1500 -T 8 --output alignment_sanitizer.fa
# ./check_alignment --raw ../data/sequences.fa --alignment alignment_sanitizer.fa
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment_sanitizer.fa
