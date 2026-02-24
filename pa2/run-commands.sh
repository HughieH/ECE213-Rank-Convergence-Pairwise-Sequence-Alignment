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

## Basic run to align the first 10 pairs (20 sequences) in the sequences.fa in batches of 2 reads
## HINT: may need to change values for the assignment tasks. You can create a sequence of commands

# Max pairs of 5000, increase batch size to 2000 compared to default of a 1000
./aligner --sequence ../data/sequences.fa --maxPairs 10 -b 2 --output ../data/baseline.fa

## Debugging with Compute Sanitizer
## Run this command to detect illegal memory accesses (out-of-bounds reads/writes) and race conditions.
# compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 1000 --output ../data/3_e.fa

## For debugging and evaluate the alignment accuracy
./check_alignment -v --raw ../data/sequences.fa --alignment ../data/baseline.fa                  # add -v to see failed result instead of a summary
./compare_alignment --reference ../data/reference_alignment.fa --estimate ../data/baseline.fa # add -v to see pair-wise comparisons
