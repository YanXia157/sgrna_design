import json

# ====== Configuration Section ======
base_dir = "./sgrna_af3_c"
file_prefix = "c_"
summary_suffix = "_summary_confidences.json"
fasta_path = "c100.fa"
max_index = 100
chain_iptm_threshold = 0.85

# ====== Collect confidences ======
confidences_list = []
name_list = []
for i in range(max_index):
    confidences_file = f"{base_dir}/{file_prefix}{i}/{file_prefix}{i}{summary_suffix}"
    try:
        with open(confidences_file, "r") as f:
            confidences = json.load(f)
            confidences_list.append(confidences)
            name_list.append(f"{file_prefix}{i}")
    except Exception:
        pass

# ====== Read FASTA file ======
seq_dict = {}
with open(fasta_path, "r") as infile:
    for line in infile:
        if line.startswith(">"):
            head = line.strip()[1:]
            seq_dict[head] = ""
        else:
            seq_dict[head] += line.strip()

# ====== Print sequences with chain_iptm > threshold ======
for i, (header, seq) in enumerate(seq_dict.items()):
    if confidences_list[i]['chain_iptm'][1] > chain_iptm_threshold:
        print(f">{name_list[i]}", confidences_list[i]['chain_iptm'][1])
        print(seq)
