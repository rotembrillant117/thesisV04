import os
import json
import random
import statistics
import sacrebleu
from sacremoses import MosesDetokenizer

DATASET_TO_LOGFILE = {
    "opus": "bleu_unprocessed.log",
    "opus_homograph": "bleu_homograph_unprocessed.log",
    "flores": "bleu_flores_unprocessed.log",
    "flores_homograph": "bleu_flores_homograph_unprocessed.log",
}


def find_relevant_log_files(lang_pair, tokenizer, dataset, base_dir="/home/brillant/thesisV04/fairseq"):
    if dataset not in DATASET_TO_LOGFILE:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Expected one of: {sorted(DATASET_TO_LOGFILE.keys())}"
        )

    tokenizer_dir = os.path.join(
        base_dir,
        f"{lang_pair}_sentencepiece_experiment_outputs",
        tokenizer,
    )

    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(
            f"Tokenizer directory does not exist: {tokenizer_dir}"
        )

    entries = os.listdir(tokenizer_dir)
    subdirs = [
        name for name in entries
        if os.path.isdir(os.path.join(tokenizer_dir, name))
    ]

    baseline_pattern = f"baseline_{lang_pair}_baseline"
    cue_pattern = f"baseline_{lang_pair}_cues"

    baseline_matches = [
        name for name in subdirs
        if baseline_pattern in name
    ]

    cue_matches = [
        name for name in subdirs
        if cue_pattern in name
    ]

    if len(baseline_matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 baseline directory in {tokenizer_dir} "
            f"matching '{baseline_pattern}', found {len(baseline_matches)}: "
            f"{baseline_matches}"
        )

    if len(cue_matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 cue directory in {tokenizer_dir} "
            f"matching '{cue_pattern}', found {len(cue_matches)}: "
            f"{cue_matches}"
        )

    baseline_dir = os.path.join(tokenizer_dir, baseline_matches[0])
    cue_dir = os.path.join(tokenizer_dir, cue_matches[0])

    log_filename = DATASET_TO_LOGFILE[dataset]

    baseline_log_path = os.path.join(baseline_dir, log_filename)
    cue_log_path = os.path.join(cue_dir, log_filename)

    if not os.path.isfile(baseline_log_path):
        raise FileNotFoundError(
            f"Baseline log file does not exist: {baseline_log_path}"
        )

    if not os.path.isfile(cue_log_path):
        raise FileNotFoundError(
            f"Cue log file does not exist: {cue_log_path}"
        )

    return baseline_log_path, cue_log_path


def parse_fairseq_log(log_path):
    result = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line:
                continue

            if line.startswith("T-"):
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Malformed T-line in {log_path}: {line}"
                    )

                sent_id_str = parts[0][2:]
                target_text = parts[1]

                try:
                    sent_id = int(sent_id_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid sentence id in T-line in {log_path}: {line}"
                    )

                if sent_id not in result:
                    result[sent_id] = (None, None)

                _, hypothesis_text = result[sent_id]
                result[sent_id] = (target_text, hypothesis_text)

            elif line.startswith("H-"):
                parts = line.split("\t", 2)
                if len(parts) != 3:
                    raise ValueError(
                        f"Malformed H-line in {log_path}: {line}"
                    )

                sent_id_str = parts[0][2:]
                hypothesis_text = parts[2]

                try:
                    sent_id = int(sent_id_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid sentence id in H-line in {log_path}: {line}"
                    )

                if sent_id not in result:
                    result[sent_id] = (None, None)

                target_text, _ = result[sent_id]
                result[sent_id] = (target_text, hypothesis_text)

            else:
                continue

    missing_targets = []
    missing_hypotheses = []

    for sent_id, (target_text, hypothesis_text) in result.items():
        if target_text is None:
            missing_targets.append(sent_id)
        if hypothesis_text is None:
            missing_hypotheses.append(sent_id)

    if missing_targets or missing_hypotheses:
        raise RuntimeError(
            f"Incomplete parsed data in {log_path}. "
            f"Missing targets for ids: {missing_targets[:10]} "
            f"(total {len(missing_targets)}). "
            f"Missing hypotheses for ids: {missing_hypotheses[:10]} "
            f"(total {len(missing_hypotheses)})."
        )

    return dict(sorted(result.items()))


def get_target_language(lang_pair):
    parts = lang_pair.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"lang_pair must look like 'en_es' or 'en_de', got: {lang_pair}"
        )
    return parts[1]


def count_unk_tokens(text):
    return text.count("<<unk>>")


def choose_reference_text(baseline_target, cue_target):
    baseline_unk_count = count_unk_tokens(baseline_target)
    cue_unk_count = count_unk_tokens(cue_target)

    if baseline_target == cue_target:
        return baseline_target

    if baseline_unk_count == 0 and cue_unk_count > 0:
        return baseline_target

    if cue_unk_count == 0 and baseline_unk_count > 0:
        return cue_target

    if baseline_unk_count < cue_unk_count:
        return baseline_target

    if cue_unk_count < baseline_unk_count:
        return cue_target

    return baseline_target


def align_system_dicts(baseline_dict, cue_dict):
    baseline_ids = set(baseline_dict.keys())
    cue_ids = set(cue_dict.keys())

    if baseline_ids != cue_ids:
        missing_in_cue = sorted(baseline_ids - cue_ids)
        missing_in_baseline = sorted(cue_ids - baseline_ids)

        raise RuntimeError(
            "Baseline and cue dictionaries do not have the same sentence ids. "
            f"Missing in cue: {missing_in_cue[:10]} (total {len(missing_in_cue)}). "
            f"Missing in baseline: {missing_in_baseline[:10]} (total {len(missing_in_baseline)})."
        )

    common_ids = sorted(baseline_ids)

    mismatched_targets = []
    for sent_id in common_ids:
        baseline_target, _ = baseline_dict[sent_id]
        cue_target, _ = cue_dict[sent_id]

        if baseline_target != cue_target:
            mismatched_targets.append(sent_id)

    if mismatched_targets:
        print(
            "Warning: baseline and cue targets do not match for some sentence ids. "
            f"First mismatched ids: {mismatched_targets[:10]} "
            f"(total {len(mismatched_targets)}). "
            "Reference text will be chosen with the following rule: "
            "if one side has <<unk>> and the other does not, use the other; "
            "if both have <<unk>>, use the one with fewer <<unk>>; "
            "otherwise use the baseline target."
        )

    return common_ids


def detokenize_lines(lines, target_lang):
    detok = MosesDetokenizer(lang=target_lang)
    detok_lines = []

    for line in lines:
        tokens = line.split()
        detok_lines.append(detok.detokenize(tokens))

    return detok_lines


def compute_bleu(hypotheses_detok, references_detok):
    bleu = sacrebleu.corpus_bleu(
        hypotheses_detok,
        [references_detok],
    )
    return bleu.score


def build_text_lists_from_ids(sent_ids, baseline_dict, cue_dict):
    references = []
    baseline_hypotheses = []
    cue_hypotheses = []

    for sent_id in sent_ids:
        baseline_target, baseline_hypothesis = baseline_dict[sent_id]
        cue_target, cue_hypothesis = cue_dict[sent_id]

        reference_text = choose_reference_text(baseline_target, cue_target)

        references.append(reference_text)
        baseline_hypotheses.append(baseline_hypothesis)
        cue_hypothheses = cue_hypothesis
        cue_hypotheses.append(cue_hypothheses)

    return references, baseline_hypotheses, cue_hypotheses


def compute_observed_scores(common_ids, baseline_dict, cue_dict, target_lang):
    references, baseline_hypotheses, cue_hypotheses = build_text_lists_from_ids(
        common_ids,
        baseline_dict,
        cue_dict,
    )

    references_detok = detokenize_lines(references, target_lang)
    baseline_hypotheses_detok = detokenize_lines(baseline_hypotheses, target_lang)
    cue_hypotheses_detok = detokenize_lines(cue_hypotheses, target_lang)

    observed_baseline_bleu = compute_bleu(baseline_hypotheses_detok, references_detok)
    observed_cue_bleu = compute_bleu(cue_hypotheses_detok, references_detok)
    observed_diff = observed_cue_bleu - observed_baseline_bleu

    return {
        "observed_baseline_bleu": observed_baseline_bleu,
        "observed_cue_bleu": observed_cue_bleu,
        "observed_diff": observed_diff,
    }


def percentile(sorted_values, p):
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list.")

    if p < 0 or p > 100:
        raise ValueError("Percentile p must be between 0 and 100.")

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (p / 100) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return lower_value + fraction * (upper_value - lower_value)


def run_paired_bootstrap(common_ids, baseline_dict, cue_dict, target_lang, num_samples=1000, seed=42):
    rng = random.Random(seed)
    results = []
    sample_size = len(common_ids)

    for iteration in range(num_samples):
        sampled_ids = [rng.choice(common_ids) for _ in range(sample_size)]

        references, baseline_hypotheses, cue_hypotheses = build_text_lists_from_ids(
            sampled_ids,
            baseline_dict,
            cue_dict,
        )

        references_detok = detokenize_lines(references, target_lang)
        baseline_hypotheses_detok = detokenize_lines(baseline_hypotheses, target_lang)
        cue_hypotheses_detok = detokenize_lines(cue_hypotheses, target_lang)

        baseline_bleu = compute_bleu(baseline_hypotheses_detok, references_detok)
        cue_bleu = compute_bleu(cue_hypotheses_detok, references_detok)
        bleu_diff = cue_bleu - baseline_bleu

        results.append({
            "iteration": iteration,
            "baseline_bleu": baseline_bleu,
            "cue_bleu": cue_bleu,
            "bleu_diff": bleu_diff,
        })

    return results


def summarize_bootstrap_results(observed_scores, bootstrap_results):
    if not bootstrap_results:
        raise ValueError("bootstrap_results is empty.")

    baseline_bleus = [row["baseline_bleu"] for row in bootstrap_results]
    cue_bleus = [row["cue_bleu"] for row in bootstrap_results]
    diffs = [row["bleu_diff"] for row in bootstrap_results]

    sorted_diffs = sorted(diffs)

    bootstrap_mean_baseline_bleu = statistics.mean(baseline_bleus)
    bootstrap_mean_cue_bleu = statistics.mean(cue_bleus)

    bootstrap_mean_diff = statistics.mean(diffs)
    bootstrap_median_diff = statistics.median(diffs)
    bootstrap_std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    ci_lower = percentile(sorted_diffs, 2.5)
    ci_upper = percentile(sorted_diffs, 97.5)

    num_positive = sum(1 for x in diffs if x > 0)
    num_negative = sum(1 for x in diffs if x < 0)
    num_zero = sum(1 for x in diffs if x == 0)

    p_cue_better = num_positive / len(diffs)
    p_baseline_better = num_negative / len(diffs)
    p_tie = num_zero / len(diffs)

    summary = {
        "observed_baseline_bleu": observed_scores["observed_baseline_bleu"],
        "observed_cue_bleu": observed_scores["observed_cue_bleu"],
        "observed_diff": observed_scores["observed_diff"],
        "bootstrap_mean_baseline_bleu": bootstrap_mean_baseline_bleu,
        "bootstrap_mean_cue_bleu": bootstrap_mean_cue_bleu,
        "bootstrap_mean_diff": bootstrap_mean_diff,
        "bootstrap_median_diff": bootstrap_median_diff,
        "bootstrap_std_diff": bootstrap_std_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_cue_better": p_cue_better,
        "p_baseline_better": p_baseline_better,
        "p_tie": p_tie,
        "num_bootstrap_samples": len(bootstrap_results),
    }

    return summary


def save_results_to_json(summary, output_dir, lang_pair, tokenizer, dataset):
    os.makedirs(output_dir, exist_ok=True)

    summary_json_path = os.path.join(
        output_dir,
        f"{lang_pair}_{tokenizer}_{dataset}_bootstrap_summary.json"
    )

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary_json_path


def run_bootstrap_experiment(
    lang_pair,
    tokenizer,
    dataset,
    num_samples=1000,
    seed=42,
    output_dir="bootstrap_results",
    base_dir="/home/brillant/thesisV04/fairseq",
):
    target_lang = get_target_language(lang_pair)

    baseline_log_path, cue_log_path = find_relevant_log_files(
        lang_pair=lang_pair,
        tokenizer=tokenizer,
        dataset=dataset,
        base_dir=base_dir,
    )

    baseline_dict = parse_fairseq_log(baseline_log_path)
    cue_dict = parse_fairseq_log(cue_log_path)

    common_ids = align_system_dicts(baseline_dict, cue_dict)

    observed_scores = compute_observed_scores(
        common_ids,
        baseline_dict,
        cue_dict,
        target_lang,
    )

    bootstrap_results = run_paired_bootstrap(
        common_ids,
        baseline_dict,
        cue_dict,
        target_lang,
        num_samples=num_samples,
        seed=seed,
    )

    summary = summarize_bootstrap_results(observed_scores, bootstrap_results)

    summary["lang_pair"] = lang_pair
    summary["tokenizer"] = tokenizer
    summary["dataset"] = dataset
    summary["target_lang"] = target_lang
    summary["num_sentences"] = len(common_ids)
    summary["baseline_log_path"] = baseline_log_path
    summary["cue_log_path"] = cue_log_path
    summary["seed"] = seed

    summary_json_path = save_results_to_json(
        summary,
        output_dir,
        lang_pair,
        tokenizer,
        dataset,
    )

    return {
        "summary": summary,
        "summary_json_path": summary_json_path,
    }


if __name__ == "__main__":
    result = run_bootstrap_experiment(
        lang_pair="en_es",
        tokenizer="bpe",
        dataset="flores",
        num_samples=1000,
        seed=42,
        output_dir="bootstrap_results",
    )

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))