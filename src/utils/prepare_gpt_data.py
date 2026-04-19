import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from src.utils.training_data_utils import get_crosslingual_homographs
from src.utils.unicode import get_language_map


OUTPUT_DIR = PROJECT_ROOT / "gpt_data"

DATASET_NAME = "Helsinki-NLP/opus-100"

# Language pairs to prepare
L2_LANGS = ["de", "es", "fr", "it", "ro", "sv"]

# OPUS-100 config names
PAIR_TO_CONFIG = {
    "de": "de-en",
    "es": "en-es",
    "fr": "en-fr",
    "it": "en-it",
    "ro": "en-ro",
    "sv": "en-sv",
}

# training_data_utils.py uses "se" for Swedish
HF_TO_HOMOGRAPH_LANG = {
    "de": "de",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "ro": "ro",
    "sv": "se",
}

EVAL_RATIO = 0.10
SEED = 42


def lowercase_text(text):
    """
    Lowercase the text.
    :param text: the text to lowercase
    :return: lowercased text
    """
    return text.lower().strip()


def inject_cues_into_text(text, homographs, cue_map):
    """
    Inject cues into text.
    :param text: the text to inject language cues
    :param homographs: homographs set
    :param cue_map: the cue map
    :return: text with language cues
    """
    parts = re.split(r'(\w+)', text)

    for i in range(len(parts)):
        word = parts[i]
        if word in homographs and len(word) > 0:
            first_char = word[0]
            replacement = cue_map.get(first_char, first_char)
            parts[i] = replacement + word[1:]

    return "".join(parts)


def split_pairs_without_sentence_overlap(pairs, eval_ratio=0.10, seed=42):
    """
    Splits the data to train and eval
    :param pairs: list of parallel sentence pairs for example, (English, German)
    :param eval_ratio: the eval set size
    :param seed: random seed
    :return:
    """
    random.seed(seed)
    random.shuffle(pairs)

    target_eval_size = int(len(pairs) * eval_ratio)

    train_pairs = []
    eval_pairs = []

    train_en_sentences = set()
    train_l2_sentences = set()
    # make sure that no eval sentences are in the training data
    for en_text, l2_text in pairs:
        can_go_to_eval = (
            len(eval_pairs) < target_eval_size
            and en_text not in train_en_sentences
            and l2_text not in train_l2_sentences
        )

        if can_go_to_eval:
            eval_pairs.append((en_text, l2_text))
        else:
            train_pairs.append((en_text, l2_text))
            train_en_sentences.add(en_text)
            train_l2_sentences.add(l2_text)

    return train_pairs, eval_pairs


def save_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def prepare_language_pair(hf_token, l2_hf):
    """
    Prepare language pair train and eval data
    :param hf_token: the huggingface token to download dataset
    :param l2_hf: the language pair
    :return:
    """
    l2_homograph = HF_TO_HOMOGRAPH_LANG[l2_hf]
    lang_pair = f"en_{l2_hf}"
    config_name = PAIR_TO_CONFIG[l2_hf]

    print(f"Preparing {lang_pair} from {DATASET_NAME} / {config_name}")

    dataset = load_dataset(
        DATASET_NAME,
        config_name,
        split="train",
        token=hf_token
    )

    homographs = get_crosslingual_homographs("en", l2_homograph)
    cues_map = get_language_map()

    en_cue_map = cues_map["en"]
    l2_cue_map = cues_map[l2_homograph]

    seen_pairs = set()
    pairs = []

    for row in dataset:
        translation = row["translation"]

        en_text = lowercase_text(translation["en"])
        l2_text = lowercase_text(translation[l2_hf])

        if not en_text or not l2_text:
            continue

        pair_key = (en_text, l2_text)
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        pairs.append((en_text, l2_text))

    train_pairs, eval_pairs = split_pairs_without_sentence_overlap(
        pairs,
        eval_ratio=EVAL_RATIO,
        seed=SEED
    )

    baseline_train_lines = []
    baseline_eval_lines = []
    cued_train_lines = []
    cued_eval_lines = []

    for en_text, l2_text in train_pairs:
        baseline_train_lines.append(en_text)
        baseline_train_lines.append(l2_text)

        cued_en = inject_cues_into_text(en_text, homographs, en_cue_map)
        cued_l2 = inject_cues_into_text(l2_text, homographs, l2_cue_map)

        cued_train_lines.append(cued_en)
        cued_train_lines.append(cued_l2)

    for en_text, l2_text in eval_pairs:
        baseline_eval_lines.append(en_text)
        baseline_eval_lines.append(l2_text)

        cued_en = inject_cues_into_text(en_text, homographs, en_cue_map)
        cued_l2 = inject_cues_into_text(l2_text, homographs, l2_cue_map)

        cued_eval_lines.append(cued_en)
        cued_eval_lines.append(cued_l2)

    pair_output_dir = OUTPUT_DIR / lang_pair
    pair_output_dir.mkdir(parents=True, exist_ok=True)

    save_lines(pair_output_dir / "baseline_train.txt", baseline_train_lines)
    save_lines(pair_output_dir / "baseline_eval.txt", baseline_eval_lines)
    save_lines(pair_output_dir / "cued_train.txt", cued_train_lines)
    save_lines(pair_output_dir / "cued_eval.txt", cued_eval_lines)

    metadata = {
        "dataset_name": DATASET_NAME,
        "config_name": config_name,
        "train_split_used": "train",
        "hf_l2_language": l2_hf,
        "homograph_l2_language": l2_homograph,
        "num_unique_pairs": len(pairs),
        "num_train_pairs": len(train_pairs),
        "num_eval_pairs": len(eval_pairs),
        "eval_ratio": EVAL_RATIO,
        "seed": SEED,
        "num_homographs": len(homographs),
    }

    with open(pair_output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done {lang_pair}: train_pairs={len(train_pairs)}, eval_pairs={len(eval_pairs)}")


def parse_args():
    """
    Parses command line arguments. Important is the huggingface token to download huggingface dataset
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Prepare GPT training/eval text corpora from OPUS-100.")
    parser.add_argument(
        "--hf_token",
        required=True,
        help="Hugging Face token used to load the dataset."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hf_token = args.hf_token

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for l2_hf in L2_LANGS:
        prepare_language_pair(hf_token, l2_hf)


if __name__ == "__main__":
    main()