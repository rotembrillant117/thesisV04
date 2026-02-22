import numpy as np
from scipy.optimize import linprog
from src.utils.training_data_utils import inject_cues
from src.stats.basic_stats import tokenization_cases
from src.stats.stats_utils import get_categories


def do_compare_stats(base_trial, sage_trial, lang, vocab_size, target_category="same_splits"):
    """
    Computes Earth Mover's Distance and homograph movement between a baseline algorithm
    and its SaGe-enhanced counterpart, saving the result to a text file.
    """
    base_name = base_trial.get_algo_name()
    sage_name = sage_trial.get_algo_name()

    # 1. Setup Trials & Data
    homographs = list(base_trial.get_homographs())
    ff_words = list(base_trial.get_ff())

    base_injected_homo = inject_cues(homographs, base_trial.get_l2())
    sage_injected_homo = inject_cues(homographs, sage_trial.get_l2())

    base_prefixed_homo = [d["cued"] for d in base_injected_homo.values()]
    sage_prefixed_homo = [d["cued"] for d in sage_injected_homo.values()]

    # Ensure we have common categories (should be same for both)
    categories = get_categories(base_trial)

    # 2. Compute Distributions (Tokenization Cases) on Homographs
    base_cases = tokenization_cases(
        base_trial.get_base_tokenizer(), base_prefixed_homo, "en", lang, categories
    )
    sage_cases = tokenization_cases(
        sage_trial.get_base_tokenizer(), sage_prefixed_homo, "en", lang, categories
    )

    # 3. Compute Metrics
    # EMD requires counts/probabilities
    base_counts = {k: len(v) for k, v in base_cases.items()}
    sage_counts = {k: len(v) for k, v in sage_cases.items()}

    emd_val, moved_dist = earth_movers_dist(categories, "en", lang, base_counts, sage_counts,
                                            track_target=target_category)

    total_mass_in_target = sum(moved_dist.values())
    moved_norm = {c: (moved_dist[c] / total_mass_in_target if total_mass_in_target > 0 else 0) for c in
                  categories}

    # Word Movements (Homographs)
    moved_to_same = words_moved_to_target(base_cases, sage_cases, categories, target_category)
    removed_from_same = words_removed_from_target(base_cases, sage_cases, categories, target_category)

    # Word Movements (False Friends)
    moved_to_same_ff = words_moved_to_target_ff(base_cases, sage_cases, ff_words, categories, target_category)

    # 4. Write Results
    # Save in the stats directory of the SAGE trial
    output_path = sage_trial.get_stats_dir() / f"comparison_vs_{base_name}.txt"

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Comparison: {base_name} vs {sage_name} ({lang}) - Vocab Size: {vocab_size}\n")
        f.write(f"Target Category: {target_category}\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Tokenization Cases (Counts on Homographs):\n")
        f.write(f"{base_name}: {base_counts}\n")
        f.write(f"{sage_name}: {sage_counts}\n\n")

        f.write(f"Earth Mover's Distance: {emd_val:.6f}\n")
        f.write(f"Total Mass Moved to Target: {total_mass_in_target:.6f}\n")
        f.write(f"Normalized Movement to Target:\n")
        for cat, val in moved_norm.items():
            if val > 0:
                f.write(f"  From {cat}: {val:.4f}\n")
        f.write("\n")

        f.write(f"Homographs Moved TO {target_category}: {sum(len(v) for v in moved_to_same.values())}\n")
        for cat, words in moved_to_same.items():
            if words:
                f.write(f"  From {cat}: {len(words)} words\n")

        f.write(f"\nHomographs Removed FROM {target_category}: {sum(len(v) for v in removed_from_same.values())}\n")
        for cat, words in removed_from_same.items():
            if words:
                f.write(f"  To {cat}: {len(words)} words\n")

        f.write(f"\nFalse Friends Moved TO {target_category}: {sum(len(v) for v in moved_to_same_ff.values())}\n")
        for cat, words in moved_to_same_ff.items():
            if words:
                f.write(f"  From {cat}: {words}\n")


def earth_movers_dist(categories, l1, l2, source, target, track_target=None):
    """
    Computes the Earth Movers Distance metric between two distributions. Also able to track how much earth was moved
     to a specific target (track_target)
    :param categories: the categories
    :param l1: English
    :param l2: other language
    :param source: source distribution
    :param target: target distribution
    :param track_target: target category to track
    :return: emd, moved
    """
    s = np.array([source[c] for c in categories], dtype=np.float64)
    t = np.array([target[c] for c in categories], dtype=np.float64)

    # Normalizing
    s /= s.sum()
    t /= t.sum()

    n = len(s)

    # Create distance matrix
    D = np.array([[dist(l1, l2, c1, c2) for c1 in categories] for c2 in categories], dtype=np.float64)
    # we are trying to minimize c.T@x where x is the solution for the linear program. So, c is the cost
    c = D.flatten()

    # Creating equality constraints
    A_eq = []
    b_eq = []

    # Supply constraints
    # [[ f00, f01, f02 ],
    # [ f10, f11, f12 ], ---> [f00, f01, f02, f10, f11, f12, f20, f21, f22]
    # [ f20, f21, f22 ]]
    # We add the row constraints. A_eq[i][j] for all j must sum to s[i]. This means we cannot move more "dirt" than we have in s[i]
    for i in range(n):
        matrix = np.zeros((n, n))
        matrix[i, :] = 1
        A_eq.append(matrix.flatten())
        b_eq.append(s[i])

    # We add more constraints. A_eq[i][j] for all i must sum to t[j]. This means we want to get exactly the amount of "dirt" at t[j]
    for j in range(n):
        matrix = np.zeros((n, n))
        matrix[:, j] = 1  # All rows in column j (incoming flows)
        A_eq.append(matrix.flatten())
        b_eq.append(t[j])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    flow_matrix = res.x.reshape((n, n))
    # Elementwise multiplication
    emd = np.sum(flow_matrix * D)
    if track_target is not None and track_target in categories:
        j = categories.index(track_target)
        moved = {categories[i]: flow_matrix[i][j] for i in range(n)}
        return emd, moved
    return emd


def dist(l1, l2, source, target):
    """
    The distance function for Earth Movers target function
    :param l1: English
    :param l2: other language
    :param source: source category
    :param target: target category
    :return:
    """
    d = {
        "same_splits": {f"{l1}_t==multi_t": 1, f"{l2}_t==multi_t": 1, f"{l1}_t=={l2}_t": 1, "different_splits": 2,
                        "same_splits": 0},
        "different_splits": {f"{l1}_t==multi_t": 1, f"{l2}_t==multi_t": 1, f"{l1}_t=={l2}_t": 1, "same_splits": 2,
                             "different_splits": 0},
        f"{l1}_t==multi_t": {f"same_splits": 1, f"{l2}_t==multi_t": 0.5, f"{l1}_t=={l2}_t": 0.7, "different_splits": 1,
                             f"{l1}_t==multi_t": 0},
        f"{l2}_t==multi_t": {f"{l1}_t==multi_t": 0.5, f"same_splits": 1, f"{l1}_t=={l2}_t": 0.7, "different_splits": 1,
                             f"{l2}_t==multi_t": 0},
        f"{l1}_t=={l2}_t": {f"{l1}_t==multi_t": 0.7, f"{l2}_t==multi_t": 0.7, f"same_splits": 1, "different_splits": 1,
                            f"{l1}_t=={l2}_t": 0}
    }

    return d[source][target]


def words_moved_to_target(num_tokens_diff1, num_tokens_diff2, categories, target):
    """
    This function checks which words moved from num_tokens_diff1 to a certain category in num_tokens_diff2
    :param num_tokens_diff1: tokenization cases 1
    :param num_tokens_diff2:tokenization cases 2
    :param categories:categories
    :param target: which words moved to target in tokenization cases 2
    :return: words moved to target
    """
    words_moved = {c: [] for c in categories}
    for c, words in num_tokens_diff1.items():
        added = set(num_tokens_diff1[c]) & set(num_tokens_diff2[target])
        for w in added:
            words_moved[c].append(w)
    return words_moved


def words_removed_from_target(num_tokens_diff1, num_tokens_diff2, categories, target):
    """
    This function checks which words moved from target in num_tokens_diff2 to other categories in num_tokens_diff1
    :param num_tokens_diff1: tokenization cases 1
    :param num_tokens_diff2: tokenization cases 2
    :param categories: categories
    :param target: which words moved out from target in tokenization cases 2
    :return: words moved out from target
    """
    words_moved = {c: [] for c in categories if c != target}
    for w in num_tokens_diff1[target]:
        if w not in set(num_tokens_diff2[target]):
            for c in words_moved.keys():
                if w in set(num_tokens_diff2[c]):
                    words_moved[c].append(w)
    return words_moved


def words_moved_to_target_ff(num_tokens_diff1, num_tokens_diff2, ff_words, categories, target):
    words_moved = words_moved_to_target(num_tokens_diff1, num_tokens_diff2, categories, target)
    ff_words_moved = {c: [] for c in categories}
    for c, words in words_moved.items():
        for ff in ff_words:
            if ff in words:
                ff_words_moved[c].append(ff)
    return ff_words_moved