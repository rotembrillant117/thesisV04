import json
import re
from collections import Counter
from pathlib import Path

from src.utils.training_data_utils import get_language_dictionary
from src.utils.unicode import get_language_map


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


def process_line(line, homograph_set, language_map, type_counter, total_counter):
    word_re = re.compile(r"(\w+)", flags=re.UNICODE)
    line = line.lower()
    parts = word_re.split(line)

    for i in range(len(parts)):
        token = parts[i]

        if token in homograph_set:
            parts[i] = mark_token(token, language_map)
            type_counter[token] += 1
            total_counter["total_marked_tokens"] += 1

    return "".join(parts)


def process_file(input_path, output_path, homograph_set, language_map, type_counter, total_counter):
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(process_line(line, homograph_set, language_map, type_counter, total_counter))


def write_stats(stats_path, lang1, lang2, initial_homograph_set, homograph_set, lang1_train_counter, lang2_train_counter, lang1_counter, lang2_counter, lang1_total, lang2_total, min_count):
    stats = {
        "languages": [lang1, lang2],
        "lowercased_corpus": True,
        "min_count_threshold_per_training_corpus_inclusive": min_count,
        "num_initial_homograph_types_len_gt_2": len(initial_homograph_set),
        "num_final_homograph_types_after_train_count_filter": len(homograph_set),
        f"{lang1}_train_homograph_counts": dict(lang1_train_counter),
        f"{lang2}_train_homograph_counts": dict(lang2_train_counter),
        lang1: {
            "num_unique_marked_words_in_corpus": len(lang1_counter),
            "num_total_marked_tokens_in_corpus": lang1_total["total_marked_tokens"],
            "marked_word_counts": dict(lang1_counter),
        },
        lang2: {
            "num_unique_marked_words_in_corpus": len(lang2_counter),
            "num_total_marked_tokens_in_corpus": lang2_total["total_marked_tokens"],
            "marked_word_counts": dict(lang2_counter),
        },
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def create_mt_cued_data(lang1, lang2, input_dir, output_dir, stats_path, min_count=5):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    stats_path = Path(stats_path)

    initial_homograph_set, homograph_set, lang1_train_counter, lang2_train_counter = build_final_homograph_set(
        lang1, lang2, input_dir, min_count=min_count
    )

    language_maps = get_language_map()
    lang1_map = language_maps[lang1]
    lang2_map = language_maps[lang2]

    output_dir.mkdir(parents=True, exist_ok=True)

    lang1_counter = Counter()
    lang2_counter = Counter()
    lang1_total = Counter()
    lang2_total = Counter()

    for split in ["train", "valid", "test"]:
        process_file(
            input_dir / f"{split}.{lang1}",
            output_dir / f"{split}.{lang1}",
            homograph_set,
            lang1_map,
            lang1_counter,
            lang1_total,
        )
        process_file(
            input_dir / f"{split}.{lang2}",
            output_dir / f"{split}.{lang2}",
            homograph_set,
            lang2_map,
            lang2_counter,
            lang2_total,
        )

    write_stats(
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
    )

    print(f"Initial homograph words (len > 2): {len(initial_homograph_set)}")
    print(f"Final homograph words after thresholding: {len(homograph_set)}")
    print(f"{lang1} unique marked words in corpus: {len(lang1_counter)}")
    print(f"{lang1} total marked tokens in corpus: {lang1_total['total_marked_tokens']}")
    print(f"{lang2} unique marked words in corpus: {len(lang2_counter)}")
    print(f"{lang2} total marked tokens in corpus: {lang2_total['total_marked_tokens']}")