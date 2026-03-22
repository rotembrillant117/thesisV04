import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from src.utils.training_data_utils import get_language_dictionary
from src.utils.unicode import get_language_map


SUPPORTED_PAIRS = {
    ("en", "de"),
    ("en", "fr"),
    ("en", "es"),
    ("en", "sv"),
    ("en", "it"),
    ("en", "ro"),
}

OPUS100_CONFIGS = {
    ("en", "de"): "de-en",
    ("en", "fr"): "en-fr",
    ("en", "es"): "en-es",
    ("en", "sv"): "en-sv",
    ("en", "it"): "en-it",
    ("en", "ro"): "en-ro",
}

FLORES_PLUS_LANG_CODES = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "sv": "swe_Latn",
    "it": "ita_Latn",
    "ro": "ron_Latn",
}


def validate_pair(lang1, lang2):
    if (lang1, lang2) not in SUPPORTED_PAIRS:
        raise ValueError(f"Unsupported pair: {(lang1, lang2)}")


def pair_prefix(lang1, lang2):
    return f"{lang1}_{lang2}"


def build_pair_paths(lang1, lang2, base_dir="fairseq/examples/homograph_translation/orig"):
    prefix = pair_prefix(lang1, lang2)
    base_dir = Path(base_dir)

    return {
        "base_dir": base_dir,
        "raw": base_dir / f"{prefix}_raw",
        "baseline": base_dir / f"{prefix}_baseline",
        "cue": base_dir / f"{prefix}_cue",
        "baseline_homograph": base_dir / f"{prefix}_baseline_homograph",
        "cue_homograph": base_dir / f"{prefix}_cue_homograph",
        "flores_raw": base_dir / f"{prefix}_flores_raw",
        "flores": base_dir / f"{prefix}_flores",
        "flores_cue": base_dir / f"{prefix}_flores_cue",
        "flores_homograph": base_dir / f"{prefix}_flores_homograph",
        "flores_cue_homograph": base_dir / f"{prefix}_flores_cue_homograph",
    }


def reset_output_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def lowercase_line(line):
    return line.lower()


def load_dataset_with_candidate_configs(dataset_name, candidate_configs, token=None):
    errors = []

    for config_name in candidate_configs:
        try:
            dataset = load_dataset(dataset_name, config_name, token=token)
            return dataset, config_name
        except Exception as e:
            errors.append((config_name, str(e)))

    message = "\n".join([f"{cfg}: {err}" for cfg, err in errors])
    raise RuntimeError(f"Could not load {dataset_name} with any candidate config.\n{message}")


def export_opus100_pair(lang1, lang2, output_dir):
    validate_pair(lang1, lang2)

    preferred = OPUS100_CONFIGS[(lang1, lang2)]
    candidate_configs = [preferred, f"{lang2}-{lang1}", f"{lang1}-{lang2}"]

    output_dir = Path(output_dir)
    reset_output_dir(output_dir)

    dataset, used_config = load_dataset_with_candidate_configs("Helsinki-NLP/opus-100", candidate_configs)

    split_name_map = {
        "train": "train",
        "validation": "valid",
        "test": "test",
    }

    for hf_split, local_split in split_name_map.items():
        src_path = output_dir / f"{local_split}.{lang1}"
        tgt_path = output_dir / f"{local_split}.{lang2}"

        with src_path.open("w", encoding="utf-8") as fsrc, tgt_path.open("w", encoding="utf-8") as ftgt:
            for example in dataset[hf_split]:
                translation = example["translation"]
                fsrc.write(translation[lang1].strip() + "\n")
                ftgt.write(translation[lang2].strip() + "\n")

    stats = {
        "dataset": "Helsinki-NLP/opus-100",
        "config_name": used_config,
        "languages": [lang1, lang2],
        "splits": {
            "train": len(dataset["train"]),
            "validation": len(dataset["validation"]),
            "test": len(dataset["test"]),
        },
        "output_dir": str(output_dir),
    }

    with (output_dir / "opus100_export_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Exported OPUS-100 {used_config} to {output_dir}")
    print(f"train={len(dataset['train'])}, validation={len(dataset['validation'])}, test={len(dataset['test'])}")


def export_flores_pair(lang1, lang2, output_dir):
    validate_pair(lang1, lang2)

    flores_lang1 = FLORES_PLUS_LANG_CODES[lang1]
    flores_lang2 = FLORES_PLUS_LANG_CODES[lang2]

    output_dir = Path(output_dir)
    reset_output_dir(output_dir)

    hf_token = os.environ.get("HF_TOKEN")

    ds_lang1 = load_dataset("openlanguagedata/flores_plus", flores_lang1, token=hf_token)
    ds_lang2 = load_dataset("openlanguagedata/flores_plus", flores_lang2, token=hf_token)

    src_path = output_dir / f"test.{lang1}"
    tgt_path = output_dir / f"test.{lang2}"

    with src_path.open("w", encoding="utf-8") as fsrc, tgt_path.open("w", encoding="utf-8") as ftgt:
        for ex1, ex2 in zip(ds_lang1["devtest"], ds_lang2["devtest"]):
            if ex1["id"] != ex2["id"]:
                raise ValueError(f"FLORES+ devtest alignment mismatch: {ex1['id']} != {ex2['id']}")
            fsrc.write(ex1["text"].strip() + "\n")
            ftgt.write(ex2["text"].strip() + "\n")

    stats = {
        "dataset": "openlanguagedata/flores_plus",
        "split_used_for_export": "devtest",
        "config_names": [flores_lang1, flores_lang2],
        "languages": [lang1, lang2],
        "flores_language_codes": {
            lang1: flores_lang1,
            lang2: flores_lang2,
        },
        "splits": {
            "dev_lang1": len(ds_lang1["dev"]),
            "devtest_lang1": len(ds_lang1["devtest"]),
            "dev_lang2": len(ds_lang2["dev"]),
            "devtest_lang2": len(ds_lang2["devtest"]),
        },
        "output_dir": str(output_dir),
    }

    with (output_dir / "flores_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Exported FLORES+ devtest {flores_lang1} + {flores_lang2} to {output_dir}")
    print(f"devtest={len(ds_lang1['devtest'])}")


def write_lowercased_file(input_path, output_path, line_counter):
    with Path(input_path).open("r", encoding="utf-8") as fin, Path(output_path).open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(lowercase_line(line))
            line_counter["num_lines"] += 1


def write_lowercase_stats(stats_path, lang1, lang2, line_counts, contains_language_cues=False, dataset_scope="all_splits"):
    stats = {
        "languages": [lang1, lang2],
        "lowercased_corpus": True,
        "contains_language_cues": contains_language_cues,
        "dataset_scope": dataset_scope,
        "line_counts": dict(line_counts),
    }

    with Path(stats_path).open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def create_mt_lowercase_data(lang1, lang2, input_dir, output_dir, stats_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    stats_path = Path(stats_path)

    reset_output_dir(output_dir)

    line_counts = {}

    for split in ["train", "valid", "test"]:
        for lang in [lang1, lang2]:
            key = f"{split}.{lang}"
            counter = {"num_lines": 0}

            write_lowercased_file(
                input_dir / f"{split}.{lang}",
                output_dir / f"{split}.{lang}",
                counter,
            )

            line_counts[key] = counter["num_lines"]

    write_lowercase_stats(
        stats_path=stats_path,
        lang1=lang1,
        lang2=lang2,
        line_counts=line_counts,
        contains_language_cues=False,
        dataset_scope="all_splits",
    )

    print("Lowercased baseline corpus created.")


def create_test_only_lowercase_data(lang1, lang2, input_dir, output_dir, stats_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    stats_path = Path(stats_path)

    reset_output_dir(output_dir)

    line_counts = {}

    for lang in [lang1, lang2]:
        key = f"test.{lang}"
        counter = {"num_lines": 0}

        write_lowercased_file(
            input_dir / f"test.{lang}",
            output_dir / f"test.{lang}",
            counter,
        )

        line_counts[key] = counter["num_lines"]

    write_lowercase_stats(
        stats_path=stats_path,
        lang1=lang1,
        lang2=lang2,
        line_counts=line_counts,
        contains_language_cues=False,
        dataset_scope="test_only",
    )


def build_initial_homograph_set(lang1, lang2):
    lang1_dict = {w.lower() for w in get_language_dictionary(lang1)}
    lang2_dict = {w.lower() for w in get_language_dictionary(lang2)}
    return {w for w in (lang1_dict & lang2_dict) if len(w) > 2}


def count_homographs_in_file(file_path, homograph_candidates):
    word_re = re.compile(r"(\w+)", flags=re.UNICODE)
    counter = Counter()

    with Path(file_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.lower()
            parts = word_re.split(line)
            for token in parts:
                if token in homograph_candidates:
                    counter[token] += 1

    return counter


def build_final_homograph_set(lang1, lang2, input_dir, min_count=5):
    input_dir = Path(input_dir)
    homograph_candidates = build_initial_homograph_set(lang1, lang2)

    lang1_train_counter = count_homographs_in_file(input_dir / f"train.{lang1}", homograph_candidates)
    lang2_train_counter = count_homographs_in_file(input_dir / f"train.{lang2}", homograph_candidates)

    final_homographs = {
        word for word in homograph_candidates
        if lang1_train_counter[word] >= min_count and lang2_train_counter[word] >= min_count
    }

    return homograph_candidates, final_homographs, lang1_train_counter, lang2_train_counter


def mark_token(token, language_map):
    if not token:
        return token

    first_char = token[0]
    if first_char not in language_map:
        return token

    return language_map[first_char] + token[1:]


def process_cued_line(line, homograph_set, language_map, type_counter, total_counter):
    word_re = re.compile(r"(\w+)", flags=re.UNICODE)
    line = line.lower()
    parts = word_re.split(line)

    for i in range(len(parts)):
        token = parts[i]

        if token in homograph_set:
            parts[i] = mark_token(token, language_map)
            type_counter[token] += 1
            total_counter["total_homographs_marked"] += 1

    return "".join(parts)


def process_cued_file(input_path, output_path, homograph_set, language_map, type_counter, total_counter):
    with Path(input_path).open("r", encoding="utf-8") as fin, Path(output_path).open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(process_cued_line(line, homograph_set, language_map, type_counter, total_counter))


def write_cued_stats(
    stats_path,
    lang1,
    lang2,
    initial_homograph_set,
    homograph_set,
    lang1_train_counter,
    lang2_train_counter,
    lang1_counter,
    lang2_counter,
    lang1_total,
    lang2_total,
    min_count,
    dataset_scope="all_splits",
):
    stats = {
        "languages": [lang1, lang2],
        "lowercased_corpus": True,
        "contains_language_cues": True,
        "dataset_scope": dataset_scope,
        "min_count_threshold_per_training_corpus_inclusive": min_count,
        "num_initial_homograph_types_len_gt_2": len(initial_homograph_set),
        "num_final_homograph_types_after_train_count_filter": len(homograph_set),
        f"{lang1}_train_homograph_counts": dict(lang1_train_counter),
        f"{lang2}_train_homograph_counts": dict(lang2_train_counter),
        lang1: {
            "num_unique_marked_words_in_corpus": len(lang1_counter),
            "total_homographs_marked": lang1_total["total_homographs_marked"],
            "marked_word_counts": dict(lang1_counter),
        },
        lang2: {
            "num_unique_marked_words_in_corpus": len(lang2_counter),
            "total_homographs_marked": lang2_total["total_homographs_marked"],
            "marked_word_counts": dict(lang2_counter),
        },
    }

    with Path(stats_path).open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def create_mt_cue_data(lang1, lang2, input_dir, output_dir, stats_path, min_count=5):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    stats_path = Path(stats_path)

    initial_homograph_set, homograph_set, lang1_train_counter, lang2_train_counter = build_final_homograph_set(
        lang1, lang2, input_dir, min_count=min_count
    )

    language_maps = get_language_map()
    lang1_map = language_maps[lang1]
    lang2_map = language_maps[lang2]

    reset_output_dir(output_dir)

    lang1_counter = Counter()
    lang2_counter = Counter()
    lang1_total = Counter()
    lang2_total = Counter()

    for split in ["train", "valid", "test"]:
        process_cued_file(
            input_dir / f"{split}.{lang1}",
            output_dir / f"{split}.{lang1}",
            homograph_set,
            lang1_map,
            lang1_counter,
            lang1_total,
        )
        process_cued_file(
            input_dir / f"{split}.{lang2}",
            output_dir / f"{split}.{lang2}",
            homograph_set,
            lang2_map,
            lang2_counter,
            lang2_total,
        )

    write_cued_stats(
        stats_path=stats_path,
        lang1=lang1,
        lang2=lang2,
        initial_homograph_set=initial_homograph_set,
        homograph_set=homograph_set,
        lang1_train_counter=lang1_train_counter,
        lang2_train_counter=lang2_train_counter,
        lang1_counter=lang1_counter,
        lang2_counter=lang2_counter,
        lang1_total=lang1_total,
        lang2_total=lang2_total,
        min_count=min_count,
        dataset_scope="all_splits",
    )

    print(f"Initial homograph words (len > 2): {len(initial_homograph_set)}")
    print(f"Final homograph words after thresholding: {len(homograph_set)}")
    print(f"{lang1} unique marked words in corpus: {len(lang1_counter)}")
    print(f"{lang1} total homographs marked in corpus: {lang1_total['total_homographs_marked']}")
    print(f"{lang2} unique marked words in corpus: {len(lang2_counter)}")
    print(f"{lang2} total homographs marked in corpus: {lang2_total['total_homographs_marked']}")


def create_test_only_cue_data(lang1, lang2, input_dir, output_dir, stats_path, reference_input_dir, min_count=5):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    stats_path = Path(stats_path)
    reference_input_dir = Path(reference_input_dir)

    initial_homograph_set, homograph_set, lang1_train_counter, lang2_train_counter = build_final_homograph_set(
        lang1, lang2, reference_input_dir, min_count=min_count
    )

    language_maps = get_language_map()
    lang1_map = language_maps[lang1]
    lang2_map = language_maps[lang2]

    reset_output_dir(output_dir)

    lang1_counter = Counter()
    lang2_counter = Counter()
    lang1_total = Counter()
    lang2_total = Counter()

    process_cued_file(
        input_dir / f"test.{lang1}",
        output_dir / f"test.{lang1}",
        homograph_set,
        lang1_map,
        lang1_counter,
        lang1_total,
    )
    process_cued_file(
        input_dir / f"test.{lang2}",
        output_dir / f"test.{lang2}",
        homograph_set,
        lang2_map,
        lang2_counter,
        lang2_total,
    )

    write_cued_stats(
        stats_path=stats_path,
        lang1=lang1,
        lang2=lang2,
        initial_homograph_set=initial_homograph_set,
        homograph_set=homograph_set,
        lang1_train_counter=lang1_train_counter,
        lang2_train_counter=lang2_train_counter,
        lang1_counter=lang1_counter,
        lang2_counter=lang2_counter,
        lang1_total=lang1_total,
        lang2_total=lang2_total,
        min_count=min_count,
        dataset_scope="test_only",
    )


def extract_words(text):
    word_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
    return word_re.findall(text.lower())


def get_homograph_only_indices(reference_src_path, final_homograph_set):
    kept_indices = []
    appeared_homograph_types = set()

    with Path(reference_src_path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            src_words = set(extract_words(line))
            matched = src_words & final_homograph_set

            if matched:
                kept_indices.append(idx)
                appeared_homograph_types.update(matched)

    return kept_indices, appeared_homograph_types


def write_filtered_file(input_path, output_path, kept_indices):
    kept_indices = set(kept_indices)

    with Path(input_path).open("r", encoding="utf-8") as fin, Path(output_path).open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx in kept_indices:
                fout.write(line)


def create_homograph_only_test_subset(
    lang1,
    lang2,
    input_dir,
    output_dir,
    reference_input_dir,
    min_count=5,
    stats_filename="homograph_stats.json",
    reference_test_source_path=None,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    reference_input_dir = Path(reference_input_dir)

    reset_output_dir(output_dir)

    _, final_homograph_set, _, _ = build_final_homograph_set(
        lang1,
        lang2,
        reference_input_dir,
        min_count=min_count,
    )

    if reference_test_source_path is None:
        reference_src = input_dir / f"test.{lang1}"
    else:
        reference_src = Path(reference_test_source_path)

    kept_indices, appeared_homograph_types = get_homograph_only_indices(reference_src, final_homograph_set)

    src_input = input_dir / f"test.{lang1}"
    tgt_input = input_dir / f"test.{lang2}"

    src_output = output_dir / f"test.{lang1}"
    tgt_output = output_dir / f"test.{lang2}"
    stats_output = output_dir / stats_filename

    write_filtered_file(src_input, src_output, kept_indices)
    write_filtered_file(tgt_input, tgt_output, kept_indices)

    total_pairs = sum(1 for _ in reference_src.open("r", encoding="utf-8"))
    kept_pairs = len(kept_indices)

    stats = {
        "lang1": lang1,
        "lang2": lang2,
        "min_count_per_language": min_count,
        "input_dir": str(input_dir),
        "reference_input_dir": str(reference_input_dir),
        "reference_test_source_path": str(reference_src),
        "output_dir": str(output_dir),
        "original_test_pairs": total_pairs,
        "kept_test_pairs": kept_pairs,
        "unique_homograph_types_in_kept_set": len(appeared_homograph_types),
        "homograph_types": sorted(appeared_homograph_types),
    }

    with stats_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Original test pairs: {total_pairs}")
    print(f"Kept test pairs: {kept_pairs}")
    print(f"Unique homograph types in kept set: {len(appeared_homograph_types)}")


def prepare_all_mt_data(lang1, lang2, base_dir="fairseq/examples/homograph_translation/orig", min_count=5):
    validate_pair(lang1, lang2)
    paths = build_pair_paths(lang1, lang2, base_dir=base_dir)

    export_opus100_pair(
        lang1=lang1,
        lang2=lang2,
        output_dir=paths["raw"],
    )

    create_mt_lowercase_data(
        lang1=lang1,
        lang2=lang2,
        input_dir=paths["raw"],
        output_dir=paths["baseline"],
        stats_path=paths["baseline"] / "baseline_stats.json",
    )

    create_mt_cue_data(
        lang1=lang1,
        lang2=lang2,
        input_dir=paths["baseline"],
        output_dir=paths["cue"],
        stats_path=paths["cue"] / "cue_stats.json",
        min_count=min_count,
    )

    create_homograph_only_test_subset(
        lang1=lang1,
        lang2=lang2,
        input_dir=paths["baseline"],
        output_dir=paths["baseline_homograph"],
        reference_input_dir=paths["baseline"],
        min_count=min_count,
        stats_filename="homograph_stats.json",
        reference_test_source_path=paths["baseline"] / f"test.{lang1}",
    )

    create_homograph_only_test_subset(
        lang1=lang1,
        lang2=lang2,
        input_dir=paths["cue"],
        output_dir=paths["cue_homograph"],
        reference_input_dir=paths["baseline"],
        min_count=min_count,
        stats_filename="homograph_stats.json",
        reference_test_source_path=paths["baseline"] / f"test.{lang1}",
    )

    export_flores_pair(
        lang1=lang1,
        lang2=lang2,
        output_dir=paths["flores_raw"],
    )

    create_test_only_lowercase_data(
        lang1=lang1,
        lang2=lang2,
        input_dir=paths["flores_raw"],
        output_dir=paths["flores"],
        stats_path=paths["flores"] / "baseline_stats.json",
    )

    create_test_only_cue_data(
        lang1=lang1,
        lang2=lang2,
        input_dir=paths["flores"],
        output_dir=paths["flores_cue"],
        stats_path=paths["flores_cue"] / "cue_stats.json",
        reference_input_dir=paths["baseline"],
        min_count=min_count,
    )

    create_homograph_only_test_subset(
        lang1=lang1,
        lang2=lang2,
        input_dir=paths["flores"],
        output_dir=paths["flores_homograph"],
        reference_input_dir=paths["baseline"],
        min_count=min_count,
        reference_test_source_path=paths["flores"] / f"test.{lang1}",
        stats_filename="homograph_stats.json",
    )

    create_homograph_only_test_subset(
        lang1=lang1,
        lang2=lang2,
        input_dir=paths["flores_cue"],
        output_dir=paths["flores_cue_homograph"],
        reference_input_dir=paths["baseline"],
        min_count=min_count,
        reference_test_source_path=paths["flores"] / f"test.{lang1}",
        stats_filename="homograph_stats.json",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang1", type=str, default="en")
    parser.add_argument("--lang2", type=str, default="de")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="fairseq/examples/homograph_translation/orig",
    )
    parser.add_argument("--min-count", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prepare_all_mt_data(
        lang1=args.lang1,
        lang2=args.lang2,
        base_dir=args.base_dir,
        min_count=args.min_count,
    )