import random

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
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def nn_score_function(self, dna):
        # calculate the score of a DNA sequence
        dna = "e" + dna + "e"
        with torch.no_grad():
            input = torch.tensor(self.tokenizer(dna)["input_ids"]).to("cuda:0")
            output = self.model(input_ids=input, labels=input)
            score = output["loss"].item()
        return math.exp(score)
    
    def log_likelihood(self, dna):
        dna = "e" + dna + "e"
        with torch.no_grad():
            input = torch.tensor(self.tokenizer(dna)["input_ids"]).to("cuda:0")
            output = self.model(input_ids=input, labels=input)
            logit = output["logits"]
            logit = logit.view(-1, logit.size(-1))
            target = input.view(-1)
            # self.cross_entropy(logit, target)
        return -self.cross_entropy(logit, target).item()

def run_nbt_test():
    from sklearn import metrics
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    df_nbt = pd.read_csv('./test_data/nbt_crispri.txt', sep='\t')

    crgpt = CRGPT('./trained_model/crgen_small/')
    pred_score = []
    for seq in [x.upper() for x in df_nbt['seq']]:
        x = crgpt.log_likelihood(seq)
        # print(x)
        pred_score.append(x)

    label = (df_nbt['label'] > 1.5).astype(int)
    pred = []
    for x in pred_score[1:]:
        if x < pred_score[0]:
            pred.append(1)
        else:
            pred.append(0)
    # print(label, pred)
    print("Accuracy", metrics.accuracy_score(label[1:], pred))
    print(metrics.classification_report(label[1:], pred, labels=[0, 1]))
    print(spearmanr(df_nbt['label'], pred_score))
    fpr, tpr, _ = metrics.roc_curve(label,  pred_score)
    auc = metrics.roc_auc_score(label, pred_score)
    plt.plot(fpr,tpr,label="CRGPT, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig("roc.png")

from sample import structure_score_function, seq_constraint

def constraint_score_function(seq):
    structure_score, mfe = structure_score_function(seq)
    seq_score = seq_constraint(seq) # 0 will pass
    nn_score = crgpt.nn_score_function(seq)
    return {
        "structure_score": structure_score,
        "constraint_mutation_score": seq_score,
        "nn_score": nn_score,
        "mfe": mfe
    }




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--seq",
    #     type=str,
    #     help="crRNA sequences",
    # )
    import sys
    import pandas as pd
    run_nbt_test()
    exit()
    exp_file = sys.argv[1]
    # df = pd.read_csv(exp_file, sep='\t', index_col=None)
    df = pd.read_csv(exp_file, sep=',', index_col=None)

    crgpt = CRGPT('./trained_model/crgen_small/')
    # print(crgpt.nn_score_function(seq))
    results = []
    for seq in df['seq']:
        res = constraint_score_function(seq)
        results.append(res)
    df_res = pd.DataFrame(results)
    df.insert(1, 'structure_score', df_res['structure_score'])
    df.insert(1, 'constraint_mutation_score', df_res['constraint_mutation_score'])
    df.insert(1, 'nn_score', df_res['nn_score'])
    df.insert(1, 'mfe', df_res['mfe'])
    df.to_csv(exp_file.replace('.csv', '_score.csv'), sep=',', index=False)