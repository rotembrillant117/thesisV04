from .tokenizer import Tokenizer
import sentencepiece as spm
from ..utils.dir_controller import MODELS_DIR
from pathlib import Path


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, language, training_corpus_dir, vocab_size, algo_name):
        self.language = language
        self.training_corpus_dir = training_corpus_dir
        self.vocab_size = vocab_size
        self.algo_name = algo_name
        self.tokenizer_name = f"{self.language}_{self.algo_name}_{vocab_size}"
        self.sp = None


    def train_tokenizer(self):
        """
        Trains the SentencePiece tokenizer and saves all artifacts (model, vocab)
        to the structured output directory.
        """

        # Determine model type
        model_type = 'unigram'  # Default
        if 'BPE' in self.algo_name:
            model_type = 'bpe'


        model_prefix = Path(MODELS_DIR / "sp" / f"{self.tokenizer_name}" / f"{self.tokenizer_name}")

        # Train
        spm.SentencePieceTrainer.Train(
            input=self.training_corpus_dir,
            model_prefix=str(model_prefix),
            vocab_size=self.vocab_size,
            model_type=model_type
        )

        # Load the trained model into memory
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_prefix.with_suffix(".model")))

    def tokenize(self, text):
        return self.sp.encode_as_pieces(text)

    def get_algo_name(self):
        return self.algo_name

    def get_training_corpus_dir(self):
        return self.training_corpus_dir

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]
