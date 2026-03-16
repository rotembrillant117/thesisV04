import json
from pathlib import Path


def lowercase_line(line):
    return line.lower()


def process_file(input_path, output_path, line_counter):
    with Path(input_path).open("r", encoding="utf-8") as fin, Path(output_path).open("w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(lowercase_line(line))
            line_counter["num_lines"] += 1


def write_stats(stats_path, lang1, lang2, line_counts):
    stats = {
        "languages": [lang1, lang2],
        "lowercased_corpus": True,
        "contains_language_cues": False,
        "line_counts": dict(line_counts),
    }

    with Path(stats_path).open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def create_mt_lowercase_data(lang1, lang2, input_dir, output_dir, stats_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    stats_path = Path(stats_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    line_counts = {}

    for split in ["train", "valid", "test"]:
        for lang in [lang1, lang2]:
            key = f"{split}.{lang}"
            counter = {"num_lines": 0}

            process_file(
                input_dir / f"{split}.{lang}",
                output_dir / f"{split}.{lang}",
                counter,
            )

            line_counts[key] = counter["num_lines"]

    write_stats(stats_path, lang1, lang2, line_counts)

    print("Lowercased baseline corpus created.")