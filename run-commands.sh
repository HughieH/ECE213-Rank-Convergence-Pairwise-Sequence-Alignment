#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

set -e

mkdir -p build
cd build
cmake ..
make -j4
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/tbb_cmake_build/tbb_cmake_build_subdir_release

# MAXPAIRS=5000
# BATCHSIZE=1000
# NUMTHREADS=8

# Normal run
#  ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1000 --numThreads 8 --output alignment.fa

# Protein run
  ./aligner --sequence ../data/subset_wol2_protein_2000.faa --maxPairs 200 --batchSize 1000 --numThreads 8 --output ../data/protein_alignment.fa
  ./check_alignment --raw ../data/subset_wol2_protein_2000.faa --alignment ../data/protein_alignment.fa               
  ./compare_alignment --reference ../data/reference_protein_alignment.fa --estimate ../data/protein_alignment.fa

# Sanitizer run
  # compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1000 --numThreads 8 --output alignment.fa

# ## Basic run to align the first 10 pairs (20 sequences) in the sequences.fa in batches of 2 reads
# ## HINT: may need to change values for the assignment tasks. You can create a sequence of commands
# ./aligner --sequence ../data/sequences.fa --maxPairs 10 --batchSize 2 --output alignment.fa ## batchSize 2

# ## Debugging with Compute Sanitizer
# ## Run this command to detect illegal memory accesses (out-of-bounds reads/writes) and race conditions.
# compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --output alignment.fa

# ## For debugging and evaluate the alignment accuracy
# ./check_alignment --raw ../data/sequences.fa --alignment alignment.fa                  # add -v to see failed result instead of a summary
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment.fa # add -v to see pair-wise comparisons
