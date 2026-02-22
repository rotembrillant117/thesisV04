import csv
from pathlib import Path
from .unicode import get_language_map

DATA_DIR = Path(Path(__file__).resolve().parent.parent.parent) / 'data'

WORDS_DATA_DIR = DATA_DIR / 'raw' /'training_data' / 'words'

LANGUAGE_DICT_DIR = DATA_DIR / 'raw' / 'all_words_in_all_languages'

def get_corpus_words(language):
    """
    Get the word frequencies of words for language in the file path. Looks at all words as lower case, so the word
    "a" and "A" are considered the same
    :param path: word frequency file path
    :return: dictionary --> {word: word_frequency}
    """
    path = Path(WORDS_DATA_DIR / language)
    word_file = [p.name for p in path.iterdir() if p.is_file()]
    path = path / word_file[0]
    word_frequencies = dict()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        word, freq = line.split("\t")[1:]
        # only lower case words
        word = word.lower()
        if word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] + int(freq.strip())
        else:
            word_frequencies[word] = int(freq.strip())
    return word_frequencies


def get_language_dictionary(language):

    languages_map = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "se": "Swedish", "it": "Italian", "ro": "Romanian"}
    # languages_map = {"en": "English", "fr": "French", "de": "German"}
    path = LANGUAGE_DICT_DIR / languages_map[language] / f"{languages_map[language]}.txt"
    with open(path, "r", encoding="utf-8") as f1:
        line1 = f1.readlines()[0].strip().lower().split(",")
    return set(line1)


def filter_words_by_frequency(word_freqs, threshold=8):
    filtered_words = {}
    for word, freq in word_freqs.items():
        if freq >= threshold:
            filtered_words[word] = freq
    return filtered_words

def filter_words_by_len(word_freqs, length=2):
    filtered_words = {}
    for word, freq in word_freqs.items():
        if len(word) > length:
            filtered_words[word] = freq
    return filtered_words

def get_crosslingual_homographs(l1, l2):
    l1_dict = get_language_dictionary(l1)
    l2_dict = get_language_dictionary(l2)
    l1_corpus_words = set(filter_words_by_len(filter_words_by_frequency(get_corpus_words(l1))).keys())
    l2_corpus_words = set(filter_words_by_len(filter_words_by_frequency(get_corpus_words(l2))).keys())
    return l1_dict & l2_dict & l1_corpus_words & l2_corpus_words



def get_ff_by_path(path):
    with open(path, 'r', encoding='utf-8') as f:
        # list of dictionaries
        return list(csv.DictReader(f))


def inject_cues(word_list, l2_lang):
    """
    Injects the appropriate language cue into a list of words.
    The language cue maps the first character to a Safe Latin character.
    Returns a dictionary of {original_word: {'cued': base_word, 'l2_cued': cued_word_l2, 'en_cued': cued_word_en}}
    """
    lang_map = get_language_map()
    en_cue_map = lang_map.get("en", {})
    l2_cue_map = lang_map.get(l2_lang, {})

    injected_words = {}

    for word in word_list:
        if not word:
            continue
        first_char = word[0]

        # 1. Base Cued
        base_cued = f"{word}"

        # 2. L2 Cued
        l2_replacement = l2_cue_map.get(first_char, first_char)
        l2_cued_word = f"{l2_replacement}{word[1:]}"

        # 3. EN Cued
        en_replacement = en_cue_map.get(first_char, first_char)
        en_cued_word = f"{en_replacement}{word[1:]}"

        injected_words[word] = {
            "cued": base_cued,
            "l2_cued": l2_cued_word,
            "en_cued": en_cued_word
        }

    return injected_words
