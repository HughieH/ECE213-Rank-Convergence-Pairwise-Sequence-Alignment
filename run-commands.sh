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

# NOTE: Un-comment out the dataset you want to check! 
# -----------------------------------------------------------------------------------------------------------------------------------------------

# -- PA2 dataset check --
./aligner --sequence ../data/sequences.fa --maxPairs 1000 --batchSize 1000 -T 8 --output ../data/pa2_alignment.fa
./check_alignment --raw ../data/sequences.fa --alignment ../data/pa2_alignment.fa
./compare_alignment --reference ../data/reference_alignment.fa --estimate ../data/pa2_alignment.fa

# -- WOL2 Protein dataset check --
# IMPORTANT: MAKE SURE to pass in the --protein flag for ./aligner & ./compare_alignment
# ./aligner --protein --sequence ../data/subset_wol2_protein_10000.faa --maxPairs 5000 --batchSize 1000 --numThreads 8 --output ../data/protein_alignment.fa
# ./check_alignment --raw ../data/subset_wol2_protein_10000.faa --alignment ../data/protein_alignment.fa               
# ./compare_alignment --protein --reference ../data/reference_protein_alignment.fa --estimate ../data/protein_alignment.fa

# -- sars_20000 dataset check --
# ./aligner --sequence ../data/sars_20000.fa --maxPairs 1000 --batchSize 700 -T 8 --output ../data/sars_alignment.fa
# ./check_alignment --raw ../data/sars_20000.fa --alignment ../data/sars_alignment.fa
# ./compare_alignment -v --reference ../data/sars_20000.msa --estimate ../data/sars_alignment.fa

# -----------------------------------------------------------------------------------------------------------------------------------------------

# ## For debugging and evaluate the alignment accuracy
# ./check_alignment --raw ../data/sequences.fa --alignment alignment.fa                  # add -v to see failed result instead of a summary
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment.fa # add -v to see pair-wise comparisons

# echo "-------- Compute Sanitizer run ------"
# compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 1500 -T 8 --output alignment_sanitizer.fa
# ./check_alignment --raw ../data/sequences.fa --alignment alignment_sanitizer.fa
# ./compare_alignment --reference ../data/reference_alignment.fa --estimate alignment_sanitizer.fa
