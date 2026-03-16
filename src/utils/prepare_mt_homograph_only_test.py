import json
import re
from pathlib import Path

from src.utils.prepare_mt_cued_data import build_final_homograph_set


def extract_words(text):
    word_re = re.compile(r"\b\w+\b", flags=re.UNICODE)
    return word_re.findall(text.lower())


def prepare_mt_homograph_only_test(
    lang1,
    lang2,
    min_count_per_language=5,
    input_dir="fairseq/examples/homograph_translation/orig/en_de_lowercase",
    output_dir="fairseq/examples/homograph_translation/orig/en_de_lowercase_homograph_only",
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, final_homograph_set, _, _ = build_final_homograph_set(
        lang1,
        lang2,
        input_dir,
        min_count=min_count_per_language,
    )

    src_input = input_dir / f"test.{lang1}"
    tgt_input = input_dir / f"test.{lang2}"

    src_output = output_dir / f"test.{lang1}"
    tgt_output = output_dir / f"test.{lang2}"
    stats_output = output_dir / "homograph_only_stats.json"

    kept_pairs = 0
    total_pairs = 0
    appeared_homograph_types = set()

    with src_input.open("r", encoding="utf-8") as fsrc_in, \
         tgt_input.open("r", encoding="utf-8") as ftgt_in, \
         src_output.open("w", encoding="utf-8") as fsrc_out, \
         tgt_output.open("w", encoding="utf-8") as ftgt_out:

        for src_line, tgt_line in zip(fsrc_in, ftgt_in):
            total_pairs += 1

            src_words = set(extract_words(src_line))
            matched = src_words & final_homograph_set

            if matched:
                fsrc_out.write(src_line)
                ftgt_out.write(tgt_line)
                kept_pairs += 1
                appeared_homograph_types.update(matched)

    stats = {
        "lang1": lang1,
        "lang2": lang2,
        "min_count_per_language": min_count_per_language,
        "input_dir": str(input_dir),
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