#!/bin/sh

# Change directory (DO NOT CHANGE!)
scriptDir=$(dirname "$(realpath "$0")")
repoDir=$(dirname "$scriptDir")   
echo $repoDir
cd $repoDir

mkdir -p build
cd build
cmake ..
make -j4
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/tbb_cmake_build/tbb_cmake_build_subdir_release

# Script to align, check, and compare score loss of sars covid DNA dataset

BATCH_SIZE=500
MAX_PAIRS=1000

# -- sars_20000 dataset check --
nsys profile -t cuda --stats=true ./aligner --sequence ../data/sars_20000.fa --maxPairs $MAX_PAIRS --batchSize $BATCH_SIZE -T 8 --output ../data/sars_alignment.fa
./check_alignment --raw ../data/sars_20000.fa --alignment ../data/sars_alignment.fa
./compare_alignment --reference ../data/sars_20000.msa --estimate ../data/sars_alignment.fa

# -----------------------------------------------------------------------------------------------------------------------------------------------

# E.g. command for kernel runtime estimates: nsys profile -t cuda --stats=true <executable>

# For debugging and evaluate the alignment accuracy
# ./check_alignment --raw ../data/sequences.fa --alignment alignment.fa                  # add -v to see failed result instead of a summary
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment.fa # add -v to see pair-wise comparisons

# echo "-------- Compute Sanitizer run ------"
# compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1500 -T 8 --output alignment_sanitizer.fa
# ./check_alignment --raw ../data/sequences.fa --alignment alignment_sanitizer.fa
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment_sanitizer.fa