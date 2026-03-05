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

# MAXPAIRS=5000
# BATCHSIZE=1000
# NUMTHREADS=8

# Normal run
#  ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1000 --numThreads 8 --output alignment.fa

# Protein run
  ./aligner --sequence ../data/subset_wol2_protein_2000.faa --maxPairs 200 --batchSize 1000 --numThreads 8 --output ../data/protein_alignment.fa
  ./check_alignment --raw ../data/subset_wol2_protein_2000.faa --alignment ../data/protein_alignment.fa               
  ./compare_alignment --reference ../data/reference_protein_alignment.fa --estimate ../data/protein_alignment.fa

# -- sars_20000 dataset check --
# ./aligner --sequence ../data/sars_20000.fa --maxPairs 200 --batchSize 150 -T 8 --output ../data/sars_alignment.fa
# ./check_alignment --raw ../data/sars_20000.fa --alignment ../data/sars_alignment.fa
# ./compare_alignment --reference ../data/sars_20000.msa --estimate ../data/sars_alignment.fa

# ## For debugging and evaluate the alignment accuracy
# ./check_alignment --raw ../data/sequences.fa --alignment alignment.fa                  # add -v to see failed result instead of a summary
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment.fa # add -v to see pair-wise comparisons



# echo "-------- Compute Sanitizer run ------"
# compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1500 -T 8 --output alignment_sanitizer.fa
# ./check_alignment --raw ../data/sequences.fa --alignment alignment_sanitizer.fa
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment_sanitizer.fa
