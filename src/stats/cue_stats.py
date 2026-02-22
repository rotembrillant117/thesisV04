from src.utils.unicode import get_language_map, get_inverse_language_map


def _write_token_distribution(lang_name, tokens, decoding_map, file_handle):
    """
    Helper to write distribution of tokens by length.
    Includes both the actual survived tokens (Safe Latin) and their readable forms (Decued).
    """
    file_handle.write(f"### Distribution of {lang_name} Cued Tokens by Length\n")
    by_len = {}
    for t in tokens:
        l = len(t)
        if l not in by_len: by_len[l] = []
        by_len[l].append(t)

    for l in sorted(by_len.keys()):
        ts = sorted(by_len[l])

        # Create readable versions
        readable_ts = []
        for t in ts:
            r = ""
            for char in t:
                r += decoding_map.get(char, char)
            readable_ts.append(r)

        file_handle.write(f"Length {l}: {len(ts)} tokens\n")
        file_handle.write(f"  Survived (Safe Latin): {ts}\n")
        file_handle.write(f"  Readable (Decued):     {readable_ts}\n")
    file_handle.write("\n")


def analyze_cue_survival(vocab, l2_lang, file_handle):
    """
    Analyzes which language cues survived in the vocabulary and their distribution across token lengths.
    """
    lang_map = get_language_map()
    inv_lang_map = get_inverse_language_map()

    # The "Safe Map" is just the expected Unicode cues (since Safe Latin == Unicode Cues)
    # Get the sets of expected cues directly from the values
    l2_expected_cues = set(lang_map.get(l2_lang, {}).values())
    en_expected_cues = set(lang_map.get("en", {}).values())

    # Decoding Maps for Readability (Unicode Cue -> ASCII)
    l2_decoding_map = inv_lang_map.get(l2_lang, {})
    en_decoding_map = inv_lang_map.get("en", {})

    l2_tokens = []
    en_tokens = []

    for token in vocab:
        has_l2 = False
        has_en = False

        # Only check if the token contains relevant cues to sort it into the distribution lists
        for char in token:
            if char in l2_expected_cues:
                has_l2 = True
            if char in en_expected_cues:
                has_en = True

        if has_l2:
            l2_tokens.append(token)
        if has_en:
            en_tokens.append(token)

    # 1. Distributions
    _write_token_distribution("English", en_tokens, en_decoding_map, file_handle)
    _write_token_distribution(l2_lang, l2_tokens, l2_decoding_map, file_handle)


def document_cue_mappings(l2_lang, file_handle):
    """
    Writes the mapping for each language cue: ASCII -> Unicode Cue.
    Includes both English and L2 for comparison.
    """
    file_handle.write(f"### Cue Mappings (English vs {l2_lang})\n")

    # Header
    # ASCII | EN Unicode Cue || L2 Unicode Cue
    header = f"{'ASCII':<5} | {'EN Unicode Cue':<20} || {f'{l2_lang} Unicode Cue':<20}"
    file_handle.write(header + "\n")
    file_handle.write("-" * len(header) + "\n")

    lang_map = get_language_map()
    l2_cue_map = lang_map.get(l2_lang, {})
    en_cue_map = lang_map.get("en", {})

    # Sort by ascii char (common keys)
    # Assumes both languages map 'a'-'z'
    sorted_ascii = sorted(l2_cue_map.keys())

    for char in sorted_ascii:
        # English Data
        en_uni = en_cue_map.get(char, "N/A")

        # L2 Data
        l2_uni = l2_cue_map.get(char, "N/A")

        # Formatting
        en_uni_fmt = f"{en_uni} ({ord(en_uni):04X})" if len(en_uni) == 1 else en_uni
        l2_uni_fmt = f"{l2_uni} ({ord(l2_uni):04X})" if len(l2_uni) == 1 else l2_uni

        char_fmt = f"'{char}'"
        row = f"{char_fmt:<5} | {en_uni_fmt:<20} || {l2_uni_fmt:<20}"
        file_handle.write(row + "\n")
    file_handle.write("\n")


def do_cue_stats(trial, vocab_size):
    """
    Main function to run cue-specific statistics.
    Writes results to 'cue_stats.txt' in the trial's stats directory.
    """
    stats_dir = trial.get_stats_dir()
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    stats_path = stats_dir / "cue_stats.txt"

    # Get False Friends
    ff_words = trial.get_ff()

    # Get Tokenizers
    tokenizers = trial.get_tokenizers()
    if len(tokenizers) <= 3 or tokenizers[3] is None:
        print(f"Warning: Missing tokenizers for {trial.algo_name} {trial.l2}.")
        return

    en_tok = tokenizers[0]
    l2_tok = tokenizers[1]
    cued_tok = tokenizers[3]

    # Get Cued Vocab
    cued_vocab = cued_tok.get_vocab()  # List[str]

    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"Cue Stats for Algo: {trial.algo_name}, Language: {trial.l2}\n")
        f.write("=" * 50 + "\n\n")

        # 1. Survival & Distribution
        analyze_cue_survival(cued_vocab, trial.l2, f)

        f.write("=" * 50 + "\n\n")

        # 3. Mappings
        document_cue_mappings(trial.l2, f)

