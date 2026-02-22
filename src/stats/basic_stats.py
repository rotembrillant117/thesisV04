import matplotlib.pyplot as plt
import math
import numpy as np


def tokenization_cases(tokenizers_list, word_list, l1, l2, categories):
    """
    This function computes an analysis on how different tokenizers split words
    :param tokenizers_list: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param word_list: list of words
    :param l1: the first language
    :param l2: the second language
    :param categories: tokenization cases
    :return: dictionary of {tokenization_case : [list of words]}
    """
    # init cases with value 0
    num_tokens_diff = {k: [] for k in categories}

    for word in word_list:
        word_tokenization = []
        num_tokens = []
        for t in tokenizers_list:
            res = t.tokenize(word)
            word_tokenization.append(res)
            num_tokens.append(len(res))
        # Same splits throughout all tokenizers
        if word_tokenization[0] == word_tokenization[2] and word_tokenization[1] == word_tokenization[2]:
            num_tokens_diff["same_splits"].append(word)
        # Same tokenization between language1 and multilingual tokenizer
        elif word_tokenization[0] == word_tokenization[2]:
            num_tokens_diff[f"{l1}_t==multi_t"].append(word)
        # Same tokenization between language2 and multilingual tokenizer
        elif word_tokenization[1] == word_tokenization[2]:
            num_tokens_diff[f"{l2}_t==multi_t"].append(word)
        # All different tokenization
        elif word_tokenization[0] != word_tokenization[1] and word_tokenization[0] != word_tokenization[2] and \
                word_tokenization[1] != word_tokenization[2]:
            num_tokens_diff["different_splits"].append(word)
        # Same tokenization between language1 and langauge2, but different from Multi tokenizer
        elif word_tokenization[0] == word_tokenization[1]:
            num_tokens_diff[f"{l1}_t=={l2}_t"].append(word)
    return num_tokens_diff


def plot_tokenization_cases(num_tokens_diff, algo, l1, l2, categories, word_types, dir):
    """
    This function plots the tokenization cases
    :param num_tokens_diff: dictionary {tokenization_case: [list of words]}
    :param algo: algo name
    :param l1: language 1
    :param l2: language 2
    :param categories: tokenization cases
    :param word_types: False Friends words or other list of words
    :param dir: directory to save graph
    :return:
    """

    plt.figure(figsize=(15, 14))
    x_axis = categories
    y_axis = [len(num_tokens_diff[key]) for key in x_axis]
    distribution = [f"{key}: {len(num_tokens_diff[key])}" for key in x_axis]
    num_words = sum(y_axis)
    fig_save_path = f"{dir}/02_token_cases_{word_types}_{l1}_{l2}_{algo}.png"
    title = f"Tokenization Cases\n{l1}, {l2}\nAlgo: {algo}\nNum words: {num_words}\nDistribution: {distribution}"
    plt.bar(x_axis, y_axis)
    plt.xticks(rotation=30, fontsize=13)
    plt.xlabel("Tokenization Splits", fontsize=15)
    plt.ylabel("Amount of Tokenization Case", fontsize=15)
    plt.title(title, fontsize=18)
    plt.savefig(fig_save_path)
    plt.close()


def get_avg_token_length_over_vocab(tokenizer):
    """
    Calculates the average token length for tokenizer vocabulary
    :param tokenizer: the tokenizer object
    :return:
    """
    vocab = tokenizer.get_vocab()
    num_chars = sum([len(v) for v in vocab])
    return num_chars / len(vocab) if len(vocab) > 0 else 0


def get_token_length_distribution(tokenizer):
    """
    Returns the token length distribution oh the tokens in the tokenizer vocabulary
    :param tokenizer: the tokenizer object
    :return:
    """
    vocab = tokenizer.get_vocab()
    distribution = dict()
    for v in vocab:
        distribution[len(v)] = distribution.get(len(v), 0) + 1
    for k, v in distribution.items():
        distribution[k] = v / len(vocab)
    sorted_dis = {key: distribution[key] for key in sorted(distribution.keys())}
    return sorted_dis


def write_tokenization_split(tokenizers, ff_data, injected_data, l2, file_handle):
    """
    Writes the tokenization splits of different tokenizers to a .txt file
    :param tokenizers: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer, cued tokenizer]
    :param ff_data: the ff data (list of words)
    :param injected_data: the dictionary containing l2_cued, en_cued, prefixed words
    :param l2: language 2
    :param algo: the algorithm used
    :param file_handle: open file handle to write to
    :return:
    """

    # Define Header
    header = f"| {'Word':<15} | {'L1 Tok':<40} | {'L2 Tok':<40} | {'Multi Tok':<40} | {'Cued (L2 cues)':<40} | {'Cued (En cues)':<40} |"
    file_handle.write(header + "\n")
    file_handle.write("|" + "-" * (len(header) - 2) + "|\n")

    for ff in ff_data:
        word = ff
        if not word:
            continue

        word_data = injected_data[word]

        # Base tokenizers
        t1 = str(tokenizers[0].tokenize(word_data["cued"]))
        t2 = str(tokenizers[1].tokenize(word_data["cued"]))
        t3 = str(tokenizers[2].tokenize(word_data["cued"]))

        cued_l2 = "N/A"
        cued_en = "N/A"

        # Cued tokenizer (4th)
        if len(tokenizers) > 3 and tokenizers[3] is not None:
            cued_tok = tokenizers[3]
            cued_l2 = str(cued_tok.tokenize(word_data["l2_cued"]))
            cued_en = str(cued_tok.tokenize(word_data["en_cued"]))

        row = f"| {word:<15} | {t1:<40} | {t2:<40} | {t3:<40} | {cued_l2:<40} | {cued_en:<40} |"
        file_handle.write(row + "\n")

    file_handle.write("\n")


def format_table(headers, rows):
    """
    Helps format a list of lists into a nice Markdown-style table.
    """
    # Initialize column widths with header lengths
    col_widths = [len(str(h)) for h in headers]

    # Update widths based on content
    for row in rows:
        for i, val in enumerate(row):
            # Format numbers to be concise if float
            s = f"{val:.4f}" if isinstance(val, float) else str(val)
            col_widths[i] = max(col_widths[i], len(s))

    # Create format string
    row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    sep_fmt = "-+-".join(["-" * w for w in col_widths])

    output = []
    output.append(row_fmt.format(*headers))
    output.append(sep_fmt)

    for row in rows:
        formatted_row = [f"{val:.4f}" if isinstance(val, float) else str(val) for val in row]
        output.append(row_fmt.format(*formatted_row))

    return "\n".join(output)

def get_fertility_distribution(tokenizer, corpus_path):
    """
    Computes a distribution of {num_subwords : count} over a training corpus.
    """
    dist = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            words = line.split()

            for word in words:
                subwords = tokenizer.tokenize(word)
                l = len(subwords)
                dist[l] = dist.get(l, 0) + 1

    return dist


def calc_renyi_efficiency(tokenizer, corpus_path, alpha=2.5):
    """
    Computes the Rényi Efficiency (alpha=2.5) of a tokenizer over its training corpus.
    Normalizes the Rényi Entropy against the maximum possible entropy (log2(V)).
    """
    token_counts = {}
    total_tokens = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            words = line.split()

            for word in words:
                subwords = tokenizer.tokenize(word)
                for sub in subwords:
                    token_counts[sub] = token_counts.get(sub, 0) + 1
                    total_tokens += 1

    V = len(token_counts)
    if V <= 1 or total_tokens == 0:
        return 0.0

    sum_p_alpha = 0.0
    for count in token_counts.values():
        p_i = count / float(total_tokens)
        sum_p_alpha += math.pow(p_i, alpha)

    h_alpha = (1.0 / (1.0 - alpha)) * math.log2(sum_p_alpha)
    h_max = math.log2(V)

    return h_alpha / h_max


def plot_fertility(multi_dist, cued_dist, trial, save_dir):
    """
    Plots a grouped bar chart for Fertility (Subwords per Word) comparing Multi and Cued tokenizers.
    """
    max_len = 7
    x = np.arange(1, max_len + 1)

    # Scale Y axis to 10 thousands
    multi_counts = [multi_dist.get(i, 0) / 10000.0 for i in x]
    cued_counts = [cued_dist.get(i, 0) / 10000.0 for i in x]

    plt.figure(figsize=(12, 8))
    width = 0.35

    plt.bar(x - width / 2, multi_counts, width, label='Multi (en_l2)', color='royalblue')
    plt.bar(x + width / 2, cued_counts, width, label='Cued (en_l2)', color='darkorange')

    plt.xlabel("Subwords in Word", fontsize=15)
    plt.ylabel("Words in Corpus (in 10,000s)", fontsize=15)

    title = f"Fertility Distribution (Multi vs Cued)\n{trial.l1}, {trial.l2} | Algo: {trial.get_algo_name()}"
    plt.title(title, fontsize=18)
    plt.xticks(x, fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    fig_save_path = save_dir / f"03_fertility_train_{trial.l1}_{trial.l2}_{trial.get_algo_name()}.png"
    plt.savefig(fig_save_path)
    plt.close()


def do_basic_stats(trial, vocab_size, injected_ff=None):
    """
    Collects basic stats and writes them to a file in the trial's stats directory.
    Includes:
    1. Comparative Table of Metrics (Vocab Stats, Train Stats, Homograph Stats)
    2. Detailed Token Length Distributions (omitted from table)
    3. Tokenization Splits (False Friends)
    """
    stats_dir = trial.get_stats_dir()
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    stats_path = stats_dir / "basic_stats.txt"

    # Collect Data First
    tokenizers = trial.get_tokenizers()

    corpus_paths = [
        trial.get_l1_corpus(),
        trial.get_l2_corpus(),
        trial.get_l1_l2_corpus(),
        trial.get_cued_corpus()
    ]

    names = ["L1 (en)", f"L2 ({trial.l2})", f"Multi (en_{trial.l2})", f"Cued (en_{trial.l2})"]
    homographs = sorted(list(trial.get_homographs()))

    table_headers = [
        "Tokenizer",
        "Avg Len (Vocab)",
        "Rényi Efficiency"
    ]

    table_rows = []

    # Dictionary to store verbose info (distributions) to print after table
    verbose_info = {}
    full_dist_info = {}  # Initialize full distribution dict

    print(f"Calculating stats for {trial.get_algo_name()}...")

    # Calculate average homograph word length (constant for trial)
    # Using length of words in characters
    avg_homograph_word_len = sum(len(h) for h in homographs) / len(homographs) if homographs else 0.0

    tokenizers_list = []  # For splitting function later

    for i in range(len(names)):
        name = names[i]
        tokenizer = tokenizers[i]
        tokenizers_list.append(tokenizer)

        # --- Basic Vocab Stats ---
        avg_len = get_avg_token_length_over_vocab(tokenizer)
        dist = get_token_length_distribution(tokenizer)
        v_size = len(tokenizer.get_vocab())

        # Store distribution for later detailed print
        # 1. Top 5 by Frequency (for summary)
        sorted_dist_freq = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        verbose_info[name] = sorted_dist_freq[:5]  # Top 5 only

        # 2. Full Distribution (Sorted by Length)
        # get_token_length_distribution already returns sorted keys (lengths)
        full_dist_info[name] = dist

        renyi_eff = calc_renyi_efficiency(tokenizer, corpus_paths[i])

        # Add to table row
        row = [
            name,
            avg_len,
            renyi_eff
        ]
        table_rows.append(row)

    # WRITE TO FILE
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Basic Stats for Algo: {trial.get_algo_name()}, L2: {trial.l2}, Vocab Size: {vocab_size}\n")
        f.write("=" * 50 + "\n\n")

        # 1. Comparative Table
        f.write("### Comparative Metrics\n")
        f.write(format_table(table_headers, table_rows))
        f.write("\n")
        f.write(f"Avg Word Len (Homographs): {avg_homograph_word_len:.4f}\n\n")

        # 2. Verbose Info (Top 5 Lengths)
        f.write("### Token Length Distributions (Top 5 by Frequency)\n")
        for name, dist in verbose_info.items():
            f.write(f"{name}: {dist}\n")
        f.write("\n")

        # 3. Full Distribution (by Length) - Table Format
        f.write("### Full Token Length Distributions (Sorted by Length)\n")

        # Get all unique lengths across all tokenizers to form the rows
        all_lengths = set()
        for dist in full_dist_info.values():
            all_lengths.update(dist.keys())
        sorted_lengths = sorted(list(all_lengths))

        # Prepare Headers
        # ["Length", "L1 (en)", "L2 (de)", ...]
        tokenizer_names = list(full_dist_info.keys())
        dist_headers = ["Length"] + tokenizer_names

        dist_rows = []
        for length in sorted_lengths:
            row = [length]
            for name in tokenizer_names:
                # get prob, default to 0 if length not in that vocab
                prob = full_dist_info[name].get(length, 0.0)
                row.append(prob)  # format_table handles rounding
            dist_rows.append(row)

        f.write(format_table(dist_headers, dist_rows))
        f.write("\n\n")

        # 3. Tokenization Splits
        f.write("=" * 50 + "\n")
        f.write("### Tokenization Splits (False Friends)\n")
        f.write("=" * 50 + "\n")

        ff_list = sorted(list(trial.get_ff()))
        # Use tokenizers_list directly
        write_tokenization_split(tokenizers_list, ff_list, injected_ff, trial.l2, f)

    # --- Generate Fertility Graph (Multi vs Cued) ---
    if len(tokenizers_list) >= 4:
        multi_tok = tokenizers_list[2]
        cued_tok = tokenizers_list[3]

        multi_corpus = trial.get_l1_l2_corpus()
        cued_corpus = trial.get_cued_corpus()

        multi_dist = get_fertility_distribution(multi_tok, multi_corpus)
        cued_dist = get_fertility_distribution(cued_tok, cued_corpus)

        plot_fertility(multi_dist, cued_dist, trial, trial.get_graphs_dir())



