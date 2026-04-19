# Tokenizing Crosslingual Homographs

This repository contains the code used for a thesis on multilingual tokenization, language cues, false friends, and downstream evaluation in machine translation.

The project has three main parts:

1. **Intrinsic tokenizer analysis**
   - train or load tokenizer trials
   - compute tokenizer statistics and comparison statistics

2. **Machine translation experiments with fairseq**
   - prepare MT data
   - train baseline and cue-based MT models
   - evaluate on multiple test sets
   - run paired bootstrap resampling over BLEU differences

3. **GPT perplexity experiments**
   - prepare multilingual English-X train/eval corpora
   - tokenize them with the already-trained SentencePiece tokenizers
   - train a small GPT model from scratch and compute perplexity

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
- `src/utils/prepare_gpt_data.py` — prepares text corpora for the GPT perplexity experiments
- `src/utils/tokenize_gpt_data.py` — tokenizes GPT corpora with existing SentencePiece models
- `src/utils/train_gpt_ppl.py` — trains small GPT models and computes perplexity
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

### 3. GPT experiments

The GPT experiments reuse the SentencePiece tokenizers trained during the intrinsic analysis (under `models/sp/`) and require a Hugging Face token for OPUS-100 data.

### 4. Python / environment

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

## Step 1: download the Hugging Face datasets

The MT data-preparation workflow uses Hugging Face datasets and requires a Hugging Face token.

Example:

```bash
export HF_TOKEN='YOUR_HF_TOKEN_HERE'
```

If the datasets are not already cached locally, set the Hugging Face token before running the preparation command.

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

---

# 3. Running the GPT perplexity experiments

The GPT workflow has 3 stages:
1. prepare multilingual English-X train/eval corpora
2. tokenize them with the already-trained SentencePiece tokenizers
3. train a small GPT model from scratch and compute perplexity

### Step 1: Prepare the GPT text corpora

```bash
python -m src.utils.prepare_gpt_data --hf_token YOUR_HF_TOKEN_HERE
```

### Step 2: Tokenize the GPT corpora with SentencePiece

```bash
python -m src.utils.tokenize_gpt_data
```

### Step 3: Train GPT and compute perplexity

Example (English–German BPE baseline):

```bash
python -m src.utils.train_gpt_ppl \
  --lang_pair en_de \
  --tokenizer_type BPE \
  --condition baseline
```

For the cued version:

```bash
python -m src.utils.train_gpt_ppl \
  --lang_pair en_de \
  --tokenizer_type BPE \
  --condition cued
```

Change `--tokenizer_type UNI` to use unigram tokenizers.

---

# 4. Example full workflow

## Tokenizer intrinsic analysis

```bash
python src/utils/prepare_training_data.py
python main.py True config/args_script3000.json
```

## MT data preparation

```bash
export HF_TOKEN='YOUR_HF_TOKEN_HERE'
python -m src.utils.prepare_all_mt_data --lang1 en --lang2 de
```

## GPT perplexity experiments

```bash
python -m src.utils.prepare_gpt_data --hf_token YOUR_HF_TOKEN_HERE
python -m src.utils.tokenize_gpt_data

python -m src.utils.train_gpt_ppl --lang_pair en_de --tokenizer_type BPE --condition baseline
python -m src.utils.train_gpt_ppl --lang_pair en_de --tokenizer_type BPE --condition cued
```

## Bootstrap significance analysis

```bash
python src/stats/bootstrap_stats2.py
```

---

# 5. Notes

- Run `src/utils/prepare_training_data.py` before the tokenizer analysis.
- Set a valid Hugging Face token before running `prepare_all_mt_data.py` or `prepare_gpt_data.py`.
- GPT experiments require the SentencePiece models from the intrinsic tokenizer analysis (`models/sp/`).
- Use Python 3.10 for the fairseq workflows to match the MT data-preparation environment used in this project.
- Each MT language pair has its own training script under:

```text
fairseq/examples/homograph_translation/
```