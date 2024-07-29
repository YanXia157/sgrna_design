import os, argparse
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2Tokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from typing import Dict
import random


class CrRnaDataset(Dataset):
    def __init__(self, tokenizer, crrnas: list, block_size=64):
        batch_encoding = tokenizer(
            crrnas, add_special_tokens=True, truncation=True, max_length=block_size
        )
        self.examples = batch_encoding["input_ids"]
        self.examples = [
            {"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


def get_tokenizer():
    tokenizer = GPT2Tokenizer("dna.json", "merges.txt")
    tokenizer.eos_token = "e"
    tokenizer.bos_token = "e"
    tokenizer.unk_token = "e"
    tokenizer.pad_token = "e"
    return tokenizer


def get_data(fasta_file: str):
    train_dna_seqs = []
    val_dna_seqs = []
    with open(fasta_file, "r") as infile:
        for line in infile:
            if line.startswith(">"):
                continue
            else:
                if random.random() > 0.9:
                    val_dna_seqs.append("e" + line.replace("\n", "").upper() + "e")
                else:
                    train_dna_seqs.append("e" + line.replace("\n", "").upper() + "e")
    return train_dna_seqs, val_dna_seqs


def get_pretrain_data(fasta_file: str):
    pretrain_dnas = ""
    tok_num = 0
    seq_num = 0
    with open(fasta_file, "r") as infile:
        for line in infile:
            if line.startswith(">"):
                continue
            else:
                if len(line) < 1000:
                    tok_num += len(line.replace("\n", ""))
                    seq_num += 1
                    pretrain_dnas += "e" + line.replace("\n", "") + "e"
    print(f"Base number: {tok_num}, Seq number: {seq_num}")
    return [pretrain_dnas[i : i + 64] for i in range(0, len(pretrain_dnas), 64)]


def get_model(length=64, size="small"):
    if size == "small":
        config = GPT2Config(
            vocab_size=8,
            n_positions=length,
            n_embd=384,
            n_layer=8,
            bos_token_id=0,
            eos_token_id=0,
        )
        model = AutoModelForCausalLM.from_config(config)
    if size == "middle":
        config = GPT2Config(
            vocab_size=8,
            n_positions=length,
            n_embd=768,
            n_layer=12,
            bos_token_id=0,
            eos_token_id=0,
        )
        model = AutoModelForCausalLM.from_config(config)
    return model


def train(
    data_path="../repeats-90_rep_seq.fasta",
    model_size="small",
    output_dir="run",
):
    output_dir = os.path.join(output_dir, f"crgen_{model_size}")
    save_dir = os.path.join(output_dir, "trained_model", f"crgen_{model_size}")
    tokenizer = get_tokenizer()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = get_model(length=64, size=model_size)
    train_dna, eval_dna = get_data(data_path)
    train_dataset = CrRnaDataset(tokenizer, train_dna)
    eval_dataset = CrRnaDataset(tokenizer, eval_dna)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=30,
        per_device_train_batch_size=32,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(save_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../repeats-90_rep_seq.fasta"
    )
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--output_dir", type=str, default="run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train(args.data_path, args.model_size, args.output_dir)
