# sgRNA Design

design sgrna handle of crispr cas9

## wildtype 61 nt

```text
gttttagagctagaaatagcaagttaaaataaggctagtccgttatcaacttgaaaaagtg
```

sites to fix:

```text
G43, G53, G27, U44, A51
```

## Setup

```bash
git clone https://github.com/bitguolab/sgrna_design.git
cd sgrna_design
gunzip repeats-90_rep_seq.fasta.gz
conda env create -f environment.yml
```

## Training data

**[CRISPRCasdb direct repeats](https://crisprcas.i2bc.paris-saclay.fr/Home/DownloadFile?filename=dr_34.zip)**, **[hmp1-II-crispr-repeats](https://github.com/biobakery/crispr2020/releases/download/v1/hmp1-II-crispr-repeats.tar.gz)**
remove blank lines in `hmp1-II-crispr-repeats` and cluster `dr_34` and `hmp1-II-crispr-repeats` using `mmseqs2` at 90% identity.


## Train

```bash
cd crgen
python cr_gen.py
```


## Run

```bash
usage: sample.py [-h] [--n_samples N_SAMPLES] [--mutation_number MUTATION_NUMBER] [--temperature TEMPERATURE]

optional arguments:
  -h, --help            show this help message and exit
  --n_samples N_SAMPLES
                        number of samples to generate
  --mutation_number MUTATION_NUMBER
                        number of mutations
  --temperature TEMPERATURE
                        temperature
```
