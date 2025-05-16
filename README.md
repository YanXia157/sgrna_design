### Workflow

1. **Prepare AF3 Input Files**
   Run `prepare_af3_inputs.py` to generate input JSON files for AlphaFold3.
   This script reads a FASTA file and a JSON template, then creates customized JSON files for each input sequence, updating sequence information and saving the results to a specified directory.

2. **Run AlphaFold3 Prediction**
   Execute AlphaFold3 using the following command (modify paths as needed):

   ```bash
   current_dir=$(pwd)
   cd <path_to_alphafold3>
   python run_alphafold.py --db_dir=<path_to_afdb> --input_dir=$current_dir --model_dir=<model_dir> --output_dir=<output_dir>
   ```

   Refer to the AlphaFold3 documentation for details on required parameters.

3. **Select Sequences Based on AF3 Confidence**
   Run `select_seq_based_on_af_confidence.py` to filter sequences according to AF3 confidence scores.
   This script reads AF3 output files, extracts confidence metrics, and selects sequences that meet the specified threshold.
