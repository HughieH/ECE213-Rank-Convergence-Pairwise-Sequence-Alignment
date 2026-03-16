# ECE 213 Project — Rank Convergence Pairwise Sequence Alignment

**Team:** Hou Wai Wan, Noah Kil, Zhao Cai

***

## Overview

This project implements GPU-accelerated pairwise sequence alignment using the **Rank Convergence (RC)** method to expose across-stage parallelism in the Needleman-Wunsch (NW) dynamic programming algorithm. Unlike standard wavefront NW which must process stages sequentially, RC splits the stage sequence into parallel chunks that start from an arbitrary frontier and iteratively converge to the correct solution, enabling significantly more GPU-friendly execution.

<img width="393" height="566" alt="Image" src="https://github.com/user-attachments/assets/da8d94d3-39de-4868-b2af-f4f94cc80f28" />

The implementation and structure of the algorithm is based on the forward pass of this LTDP pseudocode, soure: _Maleki, S., et al. Parallelizing Dynamic Programming Through Rank Convergence. PPoPP 2014._)

Both **DNA** (MATCH/MISMATCH scoring) and **protein** (BLOSUM62 scoring) alignment are supported.

The main entrypoint `./aligner` is heavily based on a modified GACT algorithm from ECE 213 PA2. It reads in a FASTA file containing multiple sequences using the kseq library (http://lh3lh3.users.sourceforge.net/kseq.shtml), transfers them to the GPU, performs pairwise alignment to compute the traceback path, and then sends the traceback path back to the host to reconstruct the aligned sequences. Our project is built around this same implementation and has a similar setup.

The `./compare_alignment` program calculates the average score loss (%) to a reference sequence alignment, and `./check_alignment` verifies that our alignment sequence is valid. 

## Setting up

Similar to how we've been doing assignments in this class, we will be using UC San Diego's Data Science/Machine Learning Platform ([DSMLP](https://blink.ucsd.edu/faculty/instruction/tech-guide/dsmlp/index.html)).

To get set up:

1. SSH into the DSMLP server (dsmlp-login.ucsd.edu) using the AD account. MacOS and Linux users can SSH into the server using the following command (replace `username` with your username)

```
ssh username@dsmlp-login.ucsd.edu
```

2. Next, clone the repo
```
cd ~
git clone https://github.com/HughieH/ECE213-Rank-Convergence-Pairwise-Sequence-Alignment.git
cd ECE213-Rank-Convergence-Pairwise-Sequence-Alignment
```

## Datasets

Our RC pairwise alignment algoirthm is capable of both DNA and Protein sequence alignment. While the protein dataset is already included in the repository, the sars DNA dataset will have be downloaded locally and scp'd onto your remote enviroment. Follow steps below to get the DNA dataset into the `data` directory of the repository.

- **Protein:** WOL2 database subset (Unaligned sequences - `data/subset_wol2_protein_10000.faa`, Reference aligned seequences - `data/reference_protein_alignment.fa`)
- **DNA:** SARS-CoV-2 sequences (Unaligned sequences - `data/sars_20000.fa`, Reference aligned seequences - `data/sars_20000.msa`) 

_Steps to download and move SARS dataset onto DSLMP:_

**Step 1)** Download `sars_20000.fa.gz` AND `sars_20000.msa.gz` from [google drive dataset](https://drive.google.com/drive/u/2/folders/13SXpD4J7mq9aHZC3CoMjWb5RXuZ4piFl) provided by professor.

**Step 2)** scp the two zipped .gz files to DSLMP, example command: `scp sars_20000.fa.gz ssh h1wan@dsmlp-login.ucsd.edu:~`. This command should scp it to your user directory. 

**Step 3)** Copy or move the .gz files to the `/data` directory in cloned repository. Use the `mv` or `cp` command.

**Step 4)** Decompress .gz file in the `data` directory using gunzip, command: `gunzip sars_20000.fa.gz`

## Different Implementations

- **Baseline 1 (CPU):** Python script using the Biopython library, used for generating reference protein sequences `./scripts/generate_reference_alignment.py`
- **Baseline 2 (GPU):** Tiled wavefront NW (`src/alignment.cu` in branch `baseline-tiling-with-protein`)
- **Rank Convergence Implementaion (GPU):** Primary implementation of RC on top of wavefront NW(`src/alignment.cu` in branch `main`)

## Running the Program

Similar to programming assignments in ECE 213, we will be using a Docker container, namely `yatisht/ece213-wi26:latest`, for submitting a job on the cluster containing the right virtual environment to build and test the code. 

1) To run protein sequence alignment on the WOL2 protein dataset:

```
/opt/launch-sh/bin/launch.sh -v a30 -c 1 -g 1 -m 8 -i yatisht/ece213-wi26:latest -f ./ECE213-Rank-Convergence-Pairwise-Sequence-Alignment/scripts/protein-align.sh
```

2) To run DNA sequence alignment on the SARS DNA dataset:

```
/opt/launch-sh/bin/launch.sh -v a30 -c 1 -g 1 -m 8 -i yatisht/ece213-wi26:latest -f ./ECE213-Rank-Convergence-Pairwise-Sequence-Alignment/scripts/sars-dna-align.sh
```

The two above scripts run the `./aligner`, `./check_alingment` and `./compare_alignment` programs in order with appropriate flags. For example, this is whats included in `protein-align.sh`:

```
# -- WOL2 Protein dataset check --
# IMPORTANT: MAKE SURE to pass in the --protein flag for ./aligner & ./compare_alignment
./aligner --protein --sequence ../data/subset_wol2_protein_10000.faa --maxPairs $MAX_PAIRS --batchSize $BATCH_SIZE --numThreads 8 --output ../data/protein_alignment.fa
./check_alignment --raw ../data/subset_wol2_protein_10000.faa --alignment ../data/protein_alignment.fa               
./compare_alignment --protein --reference ../data/reference_protein_alignment.fa --estimate ../data/protein_alignment.fa
```

