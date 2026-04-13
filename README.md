# ThesisV04

This repository contains the code used for a thesis on multilingual tokenization, language cues, false friends, and downstream evaluation in machine translation.

The project has two main parts:

1. **Intrinsic tokenizer analysis**
   - train or load tokenizer trials
   - compute tokenizer statistics and comparison statistics

2. **Machine translation experiments with fairseq**
   - prepare MT data
   - train baseline and cue-based MT models
   - evaluate on multiple test sets
   - run paired bootstrap resampling over BLEU differences

The repository entry point for the tokenizer analysis is `main.py`, which takes:
- a string flag (`True` / `False`) indicating whether to train tokenizer trials or load existing ones
- a path to a JSON config file

An example config file is available at:

```text
config/args_script3000.json
```

---

## Project structure

- `main.py` — entry point for the tokenizer intrinsic analysis
- `config/args_script3000.json` — example config file
- `src/utils/prepare_training_data.py` — prepares the training corpora used in the tokenizer experiments
- `src/utils/prepare_all_mt_data.py` — prepares the MT datasets used in the fairseq experiments
- `fairseq/examples/homograph_translation/train_en_<lang>_sentencepiece.sh` — language-pair-specific fairseq training/evaluation scripts
- `src/stats/bootstrap_stats2.py` — paired bootstrap significance analysis over MT results

---

## Prerequisites

Before running anything, make sure the required data and environments are available.

### 1. Training data for the tokenizer analysis

The tokenizer experiments require the training corpora referenced in the config file to exist.

The example config expects raw training data and word lists under paths such as:

- `./data/raw/training_data/en/...`
- `./data/raw/training_data/de/...`
- `./data/raw/training_data/es/...`
- etc.

### 2. MT data

The fairseq MT experiments require bilingual datasets to be available locally.

### 3. Python / environment

The fairseq workflows should be run in the same fairseq-compatible environment used for the MT experiments. The MT data-preparation logs in this project were run under Python 3.10, so using Python 3.10 is the safest choice for reproducing the fairseq setup.

---

# 1. Running the tokenizer intrinsic analysis

## Step 1: make sure the training data is present

Before running `main.py`, make sure the training corpora referenced in the config file exist.

Then run the training-data preparation script:

```bash
python src/utils/prepare_training_data.py
```

This script prepares the corpora used in the tokenizer experiments, including:
- cleaned training files
- lowercased corpora
- multilingual training files
- cue-marked corpora for the language pairs

## Step 2: choose a config file

Use a JSON config file such as:

```text
config/args_script3000.json
```

This example config includes:
- tokenizer algorithms
- vocabulary size
- English data
- partner-language data
- false-friend resources
- multilingual training corpora

## Step 3: run `main.py`

`main.py` expects two command-line arguments:

```bash
python main.py <True|False> <path_to_config>
```

### Train tokenizer trials and then run statistics

Use `True` if tokenizer trials should be trained:

```bash
python main.py True config/args_script3000.json
```

### Load existing tokenizer trials and only run statistics

Use `False` if the tokenizer trials already exist and only the statistics should be recomputed:

```bash
python main.py False config/args_script3000.json
```

At a high level, `main.py`:
- parses the config file
- creates directories
- trains or loads tokenizer trials
- runs the basic statistics
- runs the comparison statistics

---

# 2. Running the fairseq MT experiments

The MT workflow is:

1. download / prepare the bilingual datasets
2. preprocess them with `prepare_all_mt_data.py`
3. run the language-pair-specific fairseq training script
4. optionally run bootstrap significance testing

---

## Step 1: download the Hugging Face datasets

The MT data-preparation workflow uses Hugging Face datasets and requires a Hugging Face token.

Example:

```bash
export HF_TOKEN='YOUR_HF_TOKEN_HERE'
```

If the datasets are not already cached locally, set the Hugging Face token before running the preparation command.

---

## Step 2: run `prepare_all_mt_data.py`

Run the MT data preparation for a language pair.

Example for English–Swedish:

```bash
python -m src.utils.prepare_all_mt_data --lang1 en --lang2 sv
```

Examples for the other language pairs:

```bash
python -m src.utils.prepare_all_mt_data --lang1 en --lang2 de
python -m src.utils.prepare_all_mt_data --lang1 en --lang2 es
python -m src.utils.prepare_all_mt_data --lang1 en --lang2 fr
python -m src.utils.prepare_all_mt_data --lang1 en --lang2 it
python -m src.utils.prepare_all_mt_data --lang1 en --lang2 ro
```

This script prepares the MT datasets used by the fairseq training scripts, including:
- baseline data
- cue data
- homograph subsets
- FLORES subsets

---

## Step 3: run the fairseq training

The repository contains language-pair-specific scripts such as:

- `fairseq/examples/homograph_translation/train_en_de_sentencepiece.sh`
- `fairseq/examples/homograph_translation/train_en_es_sentencepiece.sh`
- `fairseq/examples/homograph_translation/train_en_fr_sentencepiece.sh`
- `fairseq/examples/homograph_translation/train_en_it_sentencepiece.sh`
- `fairseq/examples/homograph_translation/train_en_ro_sentencepiece.sh`
- `fairseq/examples/homograph_translation/train_en_sv_sentencepiece.sh`

These scripts:
- validate that the required MT dataset files exist
- train the tokenizer/model setup
- generate predictions
- score BLEU and chrF on:
  - main test set
  - homograph test set
  - FLORES test set
  - FLORES homograph test set

### Example: run English–Spanish baseline training

From inside:

```text
fairseq/examples/homograph_translation
```

run:

```bash
./train_en_es_sentencepiece.sh \
  --jamo-type en_es_baseline \
  --experiment-name baseline_en_es_baseline \
  --tokenizer-type bpe \
  --src-bpe-tokens 8000 \
  --tgt-bpe-tokens 8000 \
  --src-dropout 0.0 \
  --tgt-dropout 0.0 \
  --seed 0 \
  --device 0
```

### Example: run English–Spanish cue training

```bash
./train_en_es_sentencepiece.sh \
  --jamo-type en_es_cue \
  --experiment-name baseline_en_es_cues \
  --tokenizer-type bpe \
  --src-bpe-tokens 8000 \
  --tgt-bpe-tokens 8000 \
  --src-dropout 0.0 \
  --tgt-dropout 0.0 \
  --seed 0 \
  --device 0
```

The same pattern applies to the other language-pair scripts.

### What the training scripts do after generation

The language-pair scripts evaluate the trained model by:
- extracting `H` lines and `T` lines from the fairseq generation logs
- detokenizing with `sacremoses`
- computing BLEU and chrF with `sacrebleu`

For example, the scripts extract and score:
- `bleu_unprocessed.log`
- `bleu_homograph_unprocessed.log`
- `bleu_flores_unprocessed.log`
- `bleu_flores_homograph_unprocessed.log`

and then produce files such as:
- `BLEU.txt`
- `BLEU_homograph.txt`
- `BLEU_flores.txt`
- `BLEU_flores_homograph.txt`
- `CHRF.txt`
- `CHRF_homograph.txt`
- `CHRF_flores.txt`
- `CHRF_flores_homograph.txt`

---

# 3. Running the paired bootstrap significance analysis

The bootstrap script compares:

- a baseline MT model
- a cue MT model

using paired bootstrap resampling over BLEU differences.

## Script

```text
src/stats/bootstrap_stats2.py
```

## What it does

At a high level, `bootstrap_stats2.py`:

1. finds the relevant baseline and cue fairseq log files for a given:
   - language pair
   - tokenizer
   - dataset
2. parses the fairseq logs
3. aligns examples by sentence ID
4. reconstructs the reference/hypothesis text used for scoring
5. detokenizes with `sacremoses`
6. computes BLEU with `sacrebleu`
7. computes:
   - observed BLEU on the full test set
   - bootstrap BLEU differences across resampled sentence sets
8. saves a JSON summary

## How to run it

Run:

```bash
python src/stats/bootstrap_stats2.py
```

At the bottom of the script, the run configuration controls:
- which language pairs are included
- which datasets are included
- which tokenizer is used (`bpe` or `unigram`)
- number of bootstrap samples
- output directory

If needed, edit those values in the script before running it.

## Output

The script saves JSON summaries named like:

```text
<lang_pair>_<tokenizer>_<dataset>_bootstrap_summary.json
```

For example:

```text
en_es_bpe_flores_bootstrap_summary.json
```

Because the tokenizer name is included in the filename, BPE and unigram results are saved separately and do not overwrite each other.

---

# 4. Example full workflow

## Tokenizer intrinsic analysis

```bash
python src/utils/prepare_training_data.py
python main.py True config/args_script3000.json
```

If the trials already exist and only the statistics should be rerun:

```bash
python main.py False config/args_script3000.json
```

## MT data preparation

```bash
export HF_TOKEN='YOUR_HF_TOKEN_HERE'
python -m src.utils.prepare_all_mt_data --lang1 en --lang2 de
```

## Fairseq training

```bash
cd fairseq/examples/homograph_translation

./train_en_de_sentencepiece.sh \
  --jamo-type en_de_baseline \
  --experiment-name baseline_en_de_baseline \
  --tokenizer-type bpe \
  --src-bpe-tokens 8000 \
  --tgt-bpe-tokens 8000 \
  --src-dropout 0.0 \
  --tgt-dropout 0.0 \
  --seed 0 \
  --device 0
```

Then run the cue version:

```bash
./train_en_de_sentencepiece.sh \
  --jamo-type en_de_cue \
  --experiment-name baseline_en_de_cues \
  --tokenizer-type bpe \
  --src-bpe-tokens 8000 \
  --tgt-bpe-tokens 8000 \
  --src-dropout 0.0 \
  --tgt-dropout 0.0 \
  --seed 0 \
  --device 0
```

## Bootstrap significance analysis

From the repository root:

```bash
python src/stats/bootstrap_stats2.py
```

---

# 5. Notes

- Run `src/utils/prepare_training_data.py` before the tokenizer analysis.
- Set a valid Hugging Face token before running `prepare_all_mt_data.py` if the datasets are not already available locally.
- Use Python 3.10 for the fairseq workflows to match the MT data-preparation environment used in this project.
- Each MT language pair has its own training script under:

```text
fairseq/examples/homograph_translation/
```

