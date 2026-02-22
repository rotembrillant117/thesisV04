from .sage import SageTokenizer
from .sp import SentencePieceTokenizer
from ..utils.dir_controller import STATS_DIR, TRIALS_DIR
from pathlib import Path
import pickle
from ..utils.logger import setup_logger
from ..utils.training_data_utils import get_crosslingual_homographs, get_ff_by_path

logger = setup_logger(__name__)


class Trial:
    """
    Experiment object. Each experiment has 3 different tokenizers: l1, l2, l1_l2
    """

    def __init__(self, l1, l2, l1_training_corpus_dir, l2_training_corpus_dir,
                 l1_l2_training_corpus_dir, l1_l2_cues_training_corpus_dir, algo_name, vocab_size, ff_words_dir,
                 l1_tokenizer):
        # Language 1, English
        self.l1 = l1
        # Language 2, Latin text
        self.l2 = l2
        self.l1_training_corpus_dir = l1_training_corpus_dir
        self.l2_training_corpus_dir = l2_training_corpus_dir
        # Corpus that includes text from l1 and l2
        self.l1_l2_training_corpus_dir = l1_l2_training_corpus_dir
        # Corpus that includes text from l1 and l2 with cues
        self.l1_l2_cues_training_corpus_dir = l1_l2_cues_training_corpus_dir
        self.algo_name = algo_name
        self.vocab_size = vocab_size
        # The False Friends words included in l1 and l2
        self.ff_data = get_ff_by_path(ff_words_dir)
        self.l1_tokenizer = l1_tokenizer
        self.l2_tokenizer = None
        self.l1_l2_tokenizer = None
        self.l1_l2_cues_tokenizer = None
        # The directory where the trial object is saved
        self.trial_dir = Path(TRIALS_DIR / f"{self.vocab_size}" / f"{self.l2}")
        # Directory in which the trial results will be saved
        self.stats_dir = Path(STATS_DIR / f"{self.vocab_size}" / f"{self.l2}" / f"{self.algo_name}" / "stats")
        self.graphs_dir = Path(STATS_DIR / f"{self.vocab_size}" / f"{self.l2}" / f"{self.algo_name}" / "graphs")
        self.homographs = get_crosslingual_homographs(self.l1, self.l2)


    def start_trial(self):
        self._train_tokenizers()

    def get_ff(self):
        ff_words = set()
        for i in range(len(self.ff_data)):
            ff_words.add(self.ff_data[i]["False Friend"])
        return ff_words

    def get_l2(self):
        return self.l2

    def get_homographs(self):
        return self.homographs

    def get_l1_corpus(self):
        return self.l1_training_corpus_dir

    def get_l2_corpus(self):
        return self.l2_training_corpus_dir

    def get_l1_l2_corpus(self):
        return self.l1_l2_training_corpus_dir

    def get_cued_corpus(self):
        return self.l1_l2_cues_training_corpus_dir

    def get_tokenizers(self):
        return [self.l1_tokenizer, self.l2_tokenizer, self.l1_l2_tokenizer, self.l1_l2_cues_tokenizer]

    def get_base_tokenizer(self):
        return self.get_tokenizers()[:3]

    def get_cues_tokenizer(self):
        return self.l1_l2_cues_tokenizer

    def get_algo_name(self):
        return self.algo_name

    def get_vocab_size(self):
        return self.vocab_size

    def get_graphs_dir(self):
        return self.graphs_dir

    def get_stats_dir(self):
        return self.stats_dir

    def _train_tokenizers(self):
        if "SAGE" in self.algo_name:
            self.l2_tokenizer = SageTokenizer(self.l2, self.l2_training_corpus_dir, self.vocab_size, self.algo_name)
            self.l1_l2_tokenizer = SageTokenizer(f"{self.l1}_{self.l2}", self.l1_l2_training_corpus_dir,
                                                   self.vocab_size, self.algo_name)
            self.l1_l2_cues_tokenizer = SageTokenizer(f"{self.l1}_{self.l2}_cues",
                                                        self.l1_l2_cues_training_corpus_dir, self.vocab_size,
                                                        self.algo_name)
        else:
            self.l2_tokenizer = SentencePieceTokenizer(self.l2, self.l2_training_corpus_dir, self.vocab_size,
                                                       self.algo_name)
            self.l1_l2_tokenizer = SentencePieceTokenizer(f"{self.l1}_{self.l2}", self.l1_l2_training_corpus_dir,
                                                          self.vocab_size, self.algo_name)
            self.l1_l2_cues_tokenizer = SentencePieceTokenizer(f"{self.l1}_{self.l2}_cues",
                                                               self.l1_l2_cues_training_corpus_dir, self.vocab_size, self.algo_name)

        if "SAGE" in self.algo_name:
            self.l2_tokenizer.preprocess_corpus()
            self.l1_l2_tokenizer.preprocess_corpus()
            self.l1_l2_cues_tokenizer.preprocess_corpus()

        self.l2_tokenizer.train_tokenizer()
        self.l1_l2_tokenizer.train_tokenizer()
        self.l1_l2_cues_tokenizer.train_tokenizer()


    def save_trial(self):
        with open(f"{self.trial_dir}/{self.__repr__()}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_trial(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


