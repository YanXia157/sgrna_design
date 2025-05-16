import json
import os

# ====== Configuration section ======
input_json_path = "cas9_w_msa_rna_wo_msa.json"
input_fasta_path = "c100.fa"
output_dir = "sgrna_af3_c"
output_json_prefix = "C_"
output_json_suffix = ".json"
rna_prefix = "UGGUACCGAAGUUUACCGAG"
sequence_index = 2  # The index in the "sequences" list/dict to be replaced

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ====== Load JSON template ======
with open(input_json_path, "r") as f:
    af3_input_wmsa = json.load(f)

# ====== Read FASTA file ======
seq_dict = {}
with open(input_fasta_path, "r") as infile:
    for line in infile:
        if line.startswith(">"):
            head = line.strip()[1:]
            seq_dict[head] = ""
        else:
            seq_dict[head] += line.strip()

# ====== Generate new JSON files ======
for i, (header, sequence) in enumerate(seq_dict.items()):
    af3_input_wmsa["name"] = f"{output_json_prefix}{i}"
    full_seq = rna_prefix + sequence.upper().replace("T", "U")
    af3_input_wmsa["sequences"][sequence_index] = {
        "rna": {
            "id": "C",
            "sequence": full_seq,
            "modifications": [],
            "unpairedMsa": f">query\n{full_seq}\n"
        }
    }
    output_path = os.path.join(output_dir, f"{output_json_prefix}{i}{output_json_suffix}")
    with open(output_path, "w") as outfile:
        json.dump(af3_input_wmsa, outfile)
