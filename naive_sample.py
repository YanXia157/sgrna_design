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
        model_name_or_path: str = "crgen/run/crgen_small/trained_model/crgen_small/",
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="number of samples to generate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=61,
        help="number of new tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
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
    from transformers import pipeline
    import pandas as pd
    now = datetime.now()
    args = get_args()
    crgpt = CRGPT(args.path_to_model)
    # nn_score_fn = crgpt.nn_score_function
    dna = "gttttagagctagaaatagcaagttaaaataaggctagtccgttatctacttgaaaaagtg".upper()
    prompt = "e"
    n_samples = args.n_samples
    max_length = args.max_length
    temperature = args.temperature

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    generation_pipeline = pipeline("text-generation", model=crgpt.model, tokenizer=crgpt.tokenizer, device="cuda:0")
    outputs = generation_pipeline(f"e{dna[:30]}",  do_sample=True, max_new_tokens=31, return_full_text=True, num_return_sequences=args.n_samples, eos_token_id=0, pad_token_id=0)
    # outputs = generation_pipeline(prompt, max_length=max_length, num_return_sequences=n_samples, temperature=temperature, do_sample=True)
    samples = [output["generated_text"] for output in outputs]
    seqs = [seq[1:] for seq in samples]
    names = list(range(len(samples)))
    # nn_scores = [crgpt.nn_score_function(seq) for seq in seqs]

    # for i, seq in enumerate(samples):
    #     structure_distance, mfe = structure_score_function(seq[1:])
    #     print(f"gen_{i}|{structure_distance}|{mfe}: {seq}")
    results = {
        "name": names,
        "seq": seqs,
        # "nn_score": nn_scores,
    }

    df = pd.DataFrame(results)
    df.to_csv(f"{out_dir}/{now.strftime('%Y-%m-%d_%H-%M-%S')}_naive_gen_seq.csv", index=False)

    # date_time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # align_file = open(f"{out_dir}/{date_time_string}_align.ali", "w")
    # seq_file = open(f"{out_dir}/{date_time_string}_gen_seq.fasta", "w")
    # for i, seq2 in enumerate(samples):
    #     align_marker = align(dna, seq2)
    #     align_file.write(f">gen_{i}\n{dna}\n{align_marker}\n{seq2}\n")

    # for name, seq, nn_score, mfe in zip(names, samples, nn_scores, mfes):
    #     seq_file.write(f">gen_{name}|score={nn_score:.2f}|mfe={mfe:.2f}\n{seq}\n")
