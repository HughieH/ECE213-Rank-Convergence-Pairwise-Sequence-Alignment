#!/usr/bin/env python3
"""
generate_reference_alignment.py

Reads a .faa file, aligns consecutive pairs (1+2, 3+4, ...),
and writes a reference FASTA alignment file.

Link to implementation of PairwiseAligner -> https://biopython.org/docs/latest/Tutorial/chapter_pairwise.html 

Usage:
    python generate_reference_alignment.py -i input.faa -o output.faa

Example usage w/ our WOL2 protein dataset:

    python generate_reference_alignment.py -i ./data/subset_wol2_protein_10000.faa -o ./data/reference_protein_alignment.fa

NOTE: Run `pip install Bio` before running script, its a key dependancy for the script.
"""

import argparse
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices


def get_aligned_strings(alignment):
    """Extract the two aligned strings (with gaps) from a Biopython alignment."""
    aln_str = str(alignment)
    lines = aln_str.strip().split('\n')
    # Biopython formats alignment as blocks: seq1 / match-line / seq2
    # Concatenate just the sequence lines (every 3rd line starting at 0 and 2)
    seq1_parts = []
    seq2_parts = []
    for i in range(0, len(lines), 3):
        if i < len(lines):
            seq1_parts.append(lines[i].split()[-1] if lines[i].split() else '')
        if i + 2 < len(lines):
            seq2_parts.append(lines[i + 2].split()[-1] if lines[i + 2].split() else '')
    return ''.join(seq1_parts), ''.join(seq2_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  required=True, help='Input .faa file')
    parser.add_argument('-o', '--output', required=True, help='Output aligned FASTA file')
    parser.add_argument('--gap', type=float, default=-2.0,
                        help='Linear gap penalty (default: -2, matches our CUDA implementation)')
    args = parser.parse_args()

    # Configure aligner to match our scoring matrix
    aligner = PairwiseAligner()
    
    aligner.mode = 'global' # Use NW global alignment
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    # Linear gap penalty, default to -2
    aligner.open_gap_score   = args.gap
    aligner.extend_gap_score = args.gap
    aligner.target_left_open_gap_score    = args.gap
    aligner.target_left_extend_gap_score  = args.gap
    aligner.target_right_open_gap_score   = args.gap
    aligner.target_right_extend_gap_score = args.gap
    aligner.query_left_open_gap_score     = args.gap
    aligner.query_left_extend_gap_score   = args.gap
    aligner.query_right_open_gap_score    = args.gap
    aligner.query_right_extend_gap_score  = args.gap                  

    records = list(SeqIO.parse(args.input, "fasta"))

    if len(records) % 2 != 0:
        print(f"WARNING: odd number of sequences ({len(records)}), last sequence will be skipped.")

    with open(args.output, 'w') as out:
        for i in range(0, len(records) - 1, 2):
            ref = records[i]
            qry = records[i + 1]

            print(f"Aligning {ref.id} vs {qry.id} ...")
            alignments = aligner.align(str(ref.seq), str(qry.seq))
            best = alignments[0]

            # aligned() returns array of ((start,end) pairs) per sequence
            fmt = format(best, 'fasta')
            # format('fasta') gives two FASTA records: ref aligned, qry aligned
            aln_records = list(SeqIO.parse(__import__('io').StringIO(fmt), 'fasta'))

            out.write(f'>{ref.id}\n{str(aln_records[0].seq)}\n')
            out.write(f'>{qry.id}\n{str(aln_records[1].seq)}\n')

    print(f"Done. Written to {args.output}")

if __name__ == '__main__':
    main()
