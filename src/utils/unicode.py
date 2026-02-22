import unicodedata as ud
import string


def is_stable(ch):
    # Survive common normalizations
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        if ud.normalize(form, ch) != ch:
            return False
    return True


# Supported languages for cue mapping
LANGUAGES = ["en", "de", "es", "it", "se", "ro", "fr"]

def build_language_maps():
    out = {}

    # 1. Get 52 safe Latin characters (26 for EN, 26 for Universal L2)
    safe_chars = get_safe_latin_chars()
    if len(safe_chars) < 52:
        raise ValueError(f"Not enough safe Latin characters found (needed 52, got {len(safe_chars)})")

    # 2. English Set
    en_alphabet = safe_chars[0:26]
    en_map = {a: cue for a, cue in zip(string.ascii_lowercase, en_alphabet)}
    out["en"] = en_map

    # 3. Universal L2 Set
    l2_alphabet = safe_chars[26:52]
    l2_map = {a: cue for a, cue in zip(string.ascii_lowercase, l2_alphabet)}

    # Assign L2 map to all other supported languages
    for lang in LANGUAGES:
        if lang != "en":
            out[lang] = l2_map.copy()

    return out


def get_safe_latin_chars(limit=100):
    """
    Returns a list of 'limit' safe lowercase Latin characters starting from U+0180.
    Ensures characters are:
    1. Lowercase (category 'Ll')
    2. Stable under ALL normalizations (NFC, NFD, NFKC, NFKD)
    This prevents tokenizer crashes and ensures robust invertibility.
    """
    safe_chars = []
    current_cp = 0x0180

    while len(safe_chars) < limit:
        char = chr(current_cp)

        # 1. Must be Lowercase
        if ud.category(char) == 'Ll':
            # 2. Must be Stable under all forms
            if is_stable(char):
                safe_chars.append(char)

        current_cp += 1
        # Increased limit to find enough stable chars given strict filtering
        if current_cp > 0x2FFF:
            break

    return safe_chars


def get_language_map():
    return build_language_maps()


def get_inverse_language_map():
    l_map = get_language_map()
    inv_map = {}

    for lang, mapping in l_map.items():
        inv = {}
        for ascii_letter, cue_char in mapping.items():
            inv[cue_char] = ascii_letter
        inv_map[lang] = inv

    return inv_map

# LANG_CUE = build_language_maps()
# for a in "abcdefghijklmnopqrstuvwxyz":
#     print(a, "->", LANG_CUE["English"][a])

