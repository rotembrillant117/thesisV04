import numpy as np
from pathlib import Path

import sentencepiece as spm


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GPT_DATA_DIR = PROJECT_ROOT / "gpt_data"
SP_MODELS_DIR = PROJECT_ROOT / "models" / "sp"

L2_LANGS = ["de", "es", "fr", "it", "ro", "sv"]
TOKENIZER_TYPES = ["BPE", "UNI"]
VOCAB_SIZE = 8000

# gpt_data uses sv, tokenizer folders use se
GPT_TO_TOKENIZER_LANG = {
    "de": "de",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "ro": "ro",
    "sv": "se",
}


def read_lines(path):
    """
    Reads raw lines from a file
    :param path: file path
    :return: raw lines
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def get_tokenizer_model_path(l2_gpt, tokenizer_type, use_cues):
    """
    Gets the tokenizer model path
    :param l2_gpt: the language pair (German, French etc...)
    :param tokenizer_type: the tokenizer type (bpe or unigram)
    :param use_cues: baseline or cued
    :return:
    """
    l2_tok = GPT_TO_TOKENIZER_LANG[l2_gpt]

    if use_cues:
        folder_name = f"en_{l2_tok}_cues_{tokenizer_type}_{VOCAB_SIZE}"
    else:
        folder_name = f"en_{l2_tok}_{tokenizer_type}_{VOCAB_SIZE}"

    folder_path = SP_MODELS_DIR / folder_name

    model_files = list(folder_path.glob("*.model"))
    if len(model_files) != 1:
        raise ValueError(f"Expected exactly one .model file in {folder_path}, found {len(model_files)}")

    return model_files[0]


def tokenize_file(model_path, input_path):
    """
    Tokenize a file to token ids using specified model
    :param model_path: the model path
    :param input_path: the input path
    :return: tokenized sentences (ids)
    """
    lines = read_lines(input_path)
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    return sp.encode(lines, out_type=int)


def save_tokenized_npy(path, tokenized_sentences):
    """
    Save tokenized sentences
    :param path: path to save tokenized sentences
    :param tokenized_sentences: the tokenized sentences
    :return:
    """
    arr = np.array(tokenized_sentences, dtype=object)
    np.save(path, arr, allow_pickle=True)


def process_language_pair(l2_gpt):
    lang_pair = f"en_{l2_gpt}"
    pair_dir = GPT_DATA_DIR / lang_pair

    if not pair_dir.exists():
        raise ValueError(f"Missing folder: {pair_dir}")

    baseline_train_path = pair_dir / "baseline_train.txt"
    baseline_eval_path = pair_dir / "baseline_eval.txt"
    cued_train_path = pair_dir / "cued_train.txt"
    cued_eval_path = pair_dir / "cued_eval.txt"

    for path in [baseline_train_path, baseline_eval_path, cued_train_path, cued_eval_path]:
        if not path.exists():
            raise ValueError(f"Missing file: {path}")

    for tokenizer_type in TOKENIZER_TYPES:
        baseline_model_path = get_tokenizer_model_path(
            l2_gpt=l2_gpt,
            tokenizer_type=tokenizer_type,
            use_cues=False
        )

        cued_model_path = get_tokenizer_model_path(
            l2_gpt=l2_gpt,
            tokenizer_type=tokenizer_type,
            use_cues=True
        )

        baseline_train_ids = tokenize_file(baseline_model_path, baseline_train_path)
        baseline_eval_ids = tokenize_file(baseline_model_path, baseline_eval_path)

        cued_train_ids = tokenize_file(cued_model_path, cued_train_path)
        cued_eval_ids = tokenize_file(cued_model_path, cued_eval_path)

        save_tokenized_npy(
            pair_dir / f"baseline_train_{tokenizer_type}_{VOCAB_SIZE}_ids.npy",
            baseline_train_ids
        )
        save_tokenized_npy(
            pair_dir / f"baseline_eval_{tokenizer_type}_{VOCAB_SIZE}_ids.npy",
            baseline_eval_ids
        )
        save_tokenized_npy(
            pair_dir / f"cued_train_{tokenizer_type}_{VOCAB_SIZE}_ids.npy",
            cued_train_ids
        )
        save_tokenized_npy(
            pair_dir / f"cued_eval_{tokenizer_type}_{VOCAB_SIZE}_ids.npy",
            cued_eval_ids
        )

        print(f"Done {lang_pair} | {tokenizer_type}")


def main():
    for l2_gpt in L2_LANGS:
        process_language_pair(l2_gpt)


if __name__ == "__main__":
    main()