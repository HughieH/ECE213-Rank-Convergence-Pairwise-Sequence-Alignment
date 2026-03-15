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

# Script to align, check, and compare score loss of WOL2 protein dataset

BATCH_SIZE=1000
MAX_PAIRS=5000

# -- WOL2 Protein dataset check --
# IMPORTANT: MAKE SURE to pass in the --protein flag for ./aligner & ./compare_alignment
./aligner --protein --sequence ../data/subset_wol2_protein_10000.faa --maxPairs $MAX_PAIRS --batchSize $BATCH_SIZE --numThreads 8 --output ../data/protein_alignment.fa
./check_alignment --raw ../data/subset_wol2_protein_10000.faa --alignment ../data/protein_alignment.fa               
./compare_alignment --protein --reference ../data/reference_protein_alignment.fa --estimate ../data/protein_alignment.fa

# -----------------------------------------------------------------------------------------------------------------------------------------------

# E.g. command for kernel runtime estimates: nsys profile -t cuda --stats=true <executable>

# For debugging and evaluate the alignment accuracy
# ./check_alignment --raw ../data/sequences.fa --alignment alignment.fa                  # add -v to see failed result instead of a summary
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment.fa # add -v to see pair-wise comparisons

# echo "-------- Compute Sanitizer run ------"
# compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1500 -T 8 --output alignment_sanitizer.fa
# ./check_alignment --raw ../data/sequences.fa --alignment alignment_sanitizer.fa
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment_sanitizer.fa
