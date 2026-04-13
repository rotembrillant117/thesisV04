from .sp import SentencePieceTokenizer
from .sage import SageTokenizer
from .trial import Trial
import os
from ..utils.dir_controller import TRIALS_DIR
from pathlib import Path

def train_l1_tokenizers(data):
    algos = data['algos']
    vocab_size = data['vocab_size']
    l1_data = data['l1']
    l1_tokenizers = dict()
    training_corpus_dir = l1_data['training_data']
    for algo in algos:
        if "SAGE" in algo:
            tokenizer = SageTokenizer(l1_data["language"], training_corpus_dir, vocab_size, algo)
        else:
            tokenizer = SentencePieceTokenizer(l1_data["language"], training_corpus_dir, vocab_size, algo)
        if "SAGE" in algo:
            tokenizer.preprocess_corpus()
        tokenizer.train_tokenizer()
        l1_tokenizers[algo] = tokenizer
    return l1_tokenizers



def init_trials(data, l1_tokenizers):
    """
    This function instantiates the Experiment objects
    :param data: the .json data
    :param l1_tokenizers: dictionary of l1 tokenizers {algo: Tokenizer}
    :return: dictionary of {l2: [Experiment1, Experiment2...]}
    """
    experiments = dict()
    l1_data = data["l1"]
    l1 = l1_data["language"]
    l1_training_corpus_dir = l1_data["training_data"]
    l2_experiments = data["l2"]
    for l2_data in l2_experiments:
        l2 = l2_data["language"]
        experiments[l2] = []
        l2_training_corpus_dir = l2_data["training_data"]
        ff_words_path = l2_data["ff"]

        for algo in data["algos"]:
            l1_tokenizer = l1_tokenizers[algo]
            if "SAGE" in algo:
                cur_exp = Trial(l1, l2, l1_training_corpus_dir, l2_training_corpus_dir,
                                     l2_data["multilingual_training_data"], l2_data["training_data_cues"], algo,
                                     data["vocab_size"], ff_words_path, l1_tokenizer)
            else:
                cur_exp = Trial(l1, l2, l1_training_corpus_dir, l2_training_corpus_dir,
                                     l2_data["multilingual_training_data"], l2_data["training_data_cues"], algo,
                                     data["vocab_size"], ff_words_path, l1_tokenizer)
            experiments[l2].append(cur_exp)

    return experiments

def start_trials(trials):
    """
    Start the experiments
    :param experiments: dictionary of Experiment objects
    :return:
    """
    for l2, t_list in trials.items():
        for t in t_list:
            t.start_trial()

def save_trials(trials):
    """
    This functon saves the Experiment objects
    :param experiments: dictionary of Experiment objects
    :return:
    """
    for l2, t_list in trials.items():
        for t in t_list:
            t.save_trial()

def train_trials(data):
    l1_tokenizers = train_l1_tokenizers(data)
    trials = init_trials(data, l1_tokenizers)
    start_trials(trials)
    save_trials(trials)
    return trials

def load_trials(vocab_size):
    """
    This function loads the Experiment objects
    :param path: path to experiment objects
    :return: dictionary of {l2: [list of Trial objects]}
    """
    trials = dict()
    path = Path(TRIALS_DIR / f"{vocab_size}")
    for l2 in os.listdir(path):
        if not (Path(path) / l2).is_dir():
            continue
        trials[l2] = []
        for pickle_file in os.listdir(Path(path) / l2):
            trials[l2].append(Trial.load_trial(Path(path) / l2 / pickle_file))
    return trials
