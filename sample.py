import random
import RNA
from datetime import datetime

import argparse
import torch
import math
from difflib import SequenceMatcher

from transformers import GPT2Tokenizer
from transformers import AutoModelForCausalLM
from tqdm import tqdm


def align(seq1, seq2):
    """
    align two sgrna sequences
    """
    align_marker = ""
    for a, b in zip(seq1.upper(), seq2.upper()):
        if a == b:
            align_marker += "|"
        else:
            align_marker += " "
    return align_marker


def get_tokenizer():
    tokenizer = GPT2Tokenizer("crgen/dna.json", "crgen/merges.txt")
    tokenizer.eos_token = "e"
    tokenizer.bos_token = "e"
    tokenizer.unk_token = "e"
    tokenizer.pad_token = "e"
    return tokenizer


class CRGPT:
    def __init__(
        self,
        model_name_or_path: str = "trained_model/crgen_small/",
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = get_tokenizer()
        self.model.eval().to("cuda:0")

    def nn_score_function(self, dna):
        # calculate the score of a DNA sequence
        dna = "e" + dna + "e"
        with torch.no_grad():
            input = torch.tensor(self.tokenizer(dna)["input_ids"]).to("cuda:0")
            output = self.model(input_ids=input, labels=input)
            score = output["loss"].item()
        return math.exp(score)


def structure_score_function(seq):
    (structure, mfe) = RNA.fold(seq)
    # from Fig S2 in Reis et al. Nature Biotechnology. 2019.
    actual_structure = "((((((..((((....))))....))))))..(((..).)).......((((....))))."
    distance = RNA.bp_distance(structure, actual_structure)
    return distance, mfe


def mutate_dna(dna):
    # randomly choose a position in the DNA sequence to mutate
    pos = random.randint(0, len(dna) - 1)
    base = random.choice(["A", "C", "G", "T"])
    # create a mutated DNA sequence by replacing the chosen base with a random one
    mutated_dna = dna[:pos] + base + dna[pos + 1 :]

    return mutated_dna


def match_sub_string(seq1, seq2):
    matcher = SequenceMatcher(None, seq1, seq2)
    match = matcher.find_longest_match(0, len(seq1), 0, len(seq2))
    return match.size


def seq_constraint(seq):
    change = 0
    if seq[6] != "G":
        change += 1
    if seq[22] != "G":
        change += 1
    if seq[23] != "T":
        change += 1
    if seq[30] != "A":
        change += 1
    if seq[32] != "G":
        change += 1
    return change


def count_mut(seq1, seq2):
    count = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            count += 1
    return count


def sample_dna_mutation(dna, n_samples, nn_score_fn=None, mut=12, T=20, max_overlap=21):
    pbar = tqdm(total=n_samples)
    current_dna = dna
    current_structure_score, mfe = structure_score_function(dna)
    current_nn_score = nn_score_fn(dna)
    samples = [dna]
    structure_scores = [current_structure_score]
    nn_scores = [current_nn_score]
    mfes = [mfe]
    names = ["wild"]
    while len(samples) <= n_samples + 1:
        traj = [current_dna]
        current_dna = dna
        current_structure_score, mfe = structure_score_function(dna)
        mut_num = 0
        while mut_num < mut:
            mutated_dna = mutate_dna(current_dna)
            change = seq_constraint(mutated_dna)
            if mutated_dna != current_dna and change == 0:
                mutated_structure_score, mfe = structure_score_function(mutated_dna)
                mutated_nn_score = nn_score_fn(mutated_dna)
                if (
                    mutated_structure_score <= current_structure_score
                    and mutated_nn_score <= current_nn_score
                ):
                    traj.append(mutated_dna)
                    current_dna = mutated_dna
                    mut_num = count_mut(samples[0], current_dna)
                elif (
                    math.log10(mutated_structure_score / current_structure_score)
                    < 1 / T
                    and mutated_nn_score <= current_nn_score
                ):
                    traj.append(mutated_dna)
                    current_dna = mutated_dna
                    mut_num = count_mut(samples[0], current_dna)
                else:
                    current_dna = traj[-1]
            else:
                current_dna = traj[-1]
        l_max = 0
        for s in samples:
            l = match_sub_string(s, traj[-1])
            if l > l_max:
                l_max = l
        if l_max < max_overlap:
            pbar.update(1)
            samples.append(traj[-1])
            structure_scores.append(mutated_structure_score)
            nn_scores.append(mutated_nn_score)
            mfes.append(mfe)
            names.append(len(samples) - 1)
    return samples, structure_scores, nn_scores, mfes, names


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="number of samples to generate",
    )
    parser.add_argument(
        "--mutation_number",
        type=int,
        default=22,
        help="number of mutations",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=20,
        help="temperature",
    )
    parser.add_argument(
        "--max_overlap",
        type=int,
        default=21,
        help="maximum overlap",
    )
    parser.add_argument(
        "--path_to_model",
        type=str,
        default="./trained_model/crgen_small/",
        help="model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/",
        help="output directory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import os
    now = datetime.now()
    args = get_args()
    crgpt = CRGPT(args.path_to_model)
    nn_score_fn = crgpt.nn_score_function
    dna = "gttttagagctagaaatagcaagttaaaataaggctagtccgttatctacttgaaaaagtg".upper()
    n_samples = args.n_samples
    mutation_number = args.mutation_number
    temperature = args.temperature
    max_overlap = args.max_overlap
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    samples, structure_scores, nn_scores, mfes, names = sample_dna_mutation(
        dna, n_samples, nn_score_fn, mutation_number, T=temperature, max_overlap=max_overlap
    )
    date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    align_file = open(f"{out_dir}/{date_time_string}_align.ali", "w")
    seq_file = open(f"{out_dir}/{date_time_string}_gen_seq.fasta", "w")
    for i, seq2 in enumerate(samples):
        align_marker = align(dna, seq2)
        align_file.write(f">gen_{i}\n{dna}\n{align_marker}\n{seq2}\n")

    for name, seq, nn_score, mfe in zip(names, samples, nn_scores, mfes):
        seq_file.write(f">gen_{name}|score={nn_score:.2f}|mfe={mfe:.2f}\n{seq}\n")
