from .sp import SentencePieceTokenizer
from .tokenizer import Tokenizer
from ..third_party.sage_main.src.sage_tokenizer import *
from pathlib import Path
from ..utils.dir_controller import MODELS_DIR
import math
import unicodedata
import regex as re

class SageTokenizer(Tokenizer):

    def __init__(self, language, training_corpus_dir, vocab_size, algo_name):
        self.language = language
        self.training_corpus_dir = training_corpus_dir
        self.vocab_size = vocab_size
        self.algo_name = algo_name
        self.full_vocab_schedule, self.embedding_schedule = self._get_sage_schedules(vocab_size)
        self.tokenizer_name = f"{self.language}_{self.algo_name}_{vocab_size}"
        setSageFolder(Path(MODELS_DIR))
        self.initial_hexed_vocab_path = f"./models/results/{self.tokenizer_name}/initial_vocab.vocab"
        self.tokenizer = None
        self.sp_tokenizer = None
        # Updated to be {vocab_size: pruned_byte_tokens}
        self.pruned_tokens = dict()


    def _preprocess_text(self, text: str) -> str:
        """
        Mimics SentencePiece's default Text Normalizer (nmt_nfkc)
        with added support for split_by_number and split_by_unicode_script.
        """
        # 1. NFKC Normalization
        text = unicodedata.normalize("NFKC", text)

        # 2. Split by Number (insert space between Letter and Number)
        text = re.sub(r'(?<=\p{L})(?=\p{N})|(?<=\p{N})(?=\p{L})', ' ', text)

        # 3. Split by Unicode Script (insert space between Latin and Non-Latin letters)
        out = []
        for i, c in enumerate(text):
            if i > 0:
                prev_is_latin = re.match(r'\p{Latin}', text[i-1]) is not None
                curr_is_latin = re.match(r'\p{Latin}', c) is not None
                if prev_is_latin != curr_is_latin and c.isalpha() and text[i-1].isalpha():
                    out.append(' ')
            out.append(c)
        text = "".join(out)

        # 4. Whitespace compression
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

        # 5. Dummy Prepend
        if text:
            text = "\u2581" + text

        # 6. Space substitution
        text = text.replace(" ", "\u2581")

        return text

    def preprocess_corpus(self):
        """
        Preprocesses the training corpus according to SentencePiece normalization
        rules and saves it to a `.sage_preprocessed` file.
        """
        preprocessed_corpus_path = Path(str(self.training_corpus_dir) + ".sage_preprocessed")

        # Check if we already did this to avoid re-running it
        if not preprocessed_corpus_path.exists():
            with open(self.training_corpus_dir, "r", encoding="utf-8") as f_in, \
                    open(preprocessed_corpus_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    f_out.write(self._preprocess_text(line) + "\n")


    def tokenize(self, text):
        """
        Tokenizes the text and returns a list of string tokens.
        If a token is a standalone raw byte that cannot be decoded as UTF-8,
        it is represented as a hex string (e.g., '<E2>').
        """
        text = self._preprocess_text(text)
        # returns a list of ids of tokens
        token_ids = self.tokenizer.tokenize(text, tokens_only=True)
        tokens = []

        for tid in token_ids:
            # we map the ids to bytes
            raw_bytes = self.tokenizer.id_to_bytes(tid)
            try:
                # we try to decode the byte into a readable string
                tokens.append(raw_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                # if not successful, we append the hex representation of the byte
                tokens.append(f"<{raw_bytes.hex().upper()}>")

        return tokens

    def train_tokenizer(self):
        """
        Trains the tokenizer. For SaGe, we must first train a BPE/UNI tokenizer which creates an initial vocabulary
        :return:
        """
        # BPE or UNI algo
        vocab_builder_algo = self.algo_name.split("_")[0]
        # Train a BPE or UNI tokenizer to create initial vocabulary

        self.sp_tokenizer = SentencePieceTokenizer(self.language, self.training_corpus_dir, self.full_vocab_schedule[0],
                                                   vocab_builder_algo)
        self.sp_tokenizer.train_tokenizer()
        vocab = sorted(self.sp_tokenizer.get_vocab())
        # Turn the vocabulary from letters to hexadecimal format, and add certain tokens that might be missing
        hexed_vocab = self._add_single_bytes(self._hex_vocab(vocab))
        max_len = max([len(bytes.fromhex(str(v))) for v in hexed_vocab])

        # Save the hexed vocabulary to a .vocab file
        with open(self.initial_hexed_vocab_path, 'w', encoding='utf-8') as vocab_file:
            for hexed_v in hexed_vocab:
                vocab_file.write(f"{hexed_v}\n")

        # Create a preprocessed corpus file path (assumes preprocess_corpus() was called upstream)
        preprocessed_corpus_path = Path(str(self.training_corpus_dir) + ".sage_preprocessed")

        # Build the SaGe vocabulary
        trainer = SaGeVocabBuilder(full_vocab_schedule=self.full_vocab_schedule,
                                   embeddings_schedule=self.embedding_schedule,
                                   workers_number=6, max_len=max_len)

        # Use the preprocessed corpus instead
        trainer.build_vocab(experiment_name=self.tokenizer_name, corpus_filepath=preprocessed_corpus_path,
                            pruned_tokens=self.pruned_tokens,
                            vocabulary_filepath=self.initial_hexed_vocab_path)
        self.save_pruned_tokens(f"./models/results/{self.tokenizer_name}/pruned_tokens.txt")
        # The final SaGe vocab is saved to a .vocab file in a certain path. Opens the file and turns it to bytes format for SaGeTokenizer object
        with open(self._get_final_vocab_path(), "r") as f:
            initial_vocab = [bytes.fromhex(line.strip()) for line in f]

        # Calculate max_len from the loaded vocabulary to ensure correct tokenization
        final_max_len = max([len(token) for token in initial_vocab]) if initial_vocab else 16
        tokenizer = SaGeTokenizer(initial_vocabulary=initial_vocab, max_len=final_max_len)

        self.tokenizer = tokenizer
        self.save_readable_vocab()

    def _hex_vocab(self, vocab):
        """
        Translates the SaGE vocabulary to hexadecimal format
        :param vocab: list of vocabulary words generated by BPE or UNI or other tokenizers
        :return: list of hexadecimal vocabulary
        """
        hexed_vocab = []
        for v in vocab:
            if isinstance(v, str):
                hex_token = v.encode("utf-8").hex()
                hexed_vocab.append(hex_token)
        return hexed_vocab

    def _add_single_bytes(self, vocab):
        """
        SaGe requires all single bytes to be in the vocabulary. This function adds them in to the vocabulary in hexadecimal
        format, if needed
        :param vocab: list of hexadecimal vocabulary
        :return: updated vocabulary
        """
        for i in range(256):
            t = f"{i:02x}"
            if t not in vocab:
                vocab.append(t)
        return vocab

    def save_pruned_tokens(self, path):
        """
        Saves the pruned tokens
        :param path: path to save the pruned tokens
        :return:
        """
        with open(path, "w") as f:
            for cur_vocab_size, byte_tokens_list in self.pruned_tokens.items():
                f.write(f"Tokens pruned at vocab size {cur_vocab_size}\n")
                for byte_token in byte_tokens_list:
                    t = byte_token.decode("utf-8", errors="replace")
                    f.write(f"{t}\n")

    def save_readable_vocab(self):
        """
        Saves the final readable vocabulary to a .vocab file in the results folder.
        The vocabulary is saved as one token per line in readable string format,
        sorted by length and then alphabetically.
        """
        path = f"./models/results/{self.tokenizer_name}/final_readable_vocab.vocab"
        vocab = self.get_vocab()

        # Sort by length (asc) then by alphabet (asc)
        sorted_vocab = sorted(vocab, key=lambda x: (len(x), x))

        with open(path, "w", encoding="utf-8") as f:
            for token in sorted_vocab:
                f.write(f"{token}\n")

    def _get_sage_schedules(self, target_vocab_size):
        """
        Private method mimicking the SaGe schedules from tktkt.
        Returns: (vocabulary_points, recompute_embeddings_at)
        """

        # Vocabulary Bounds (Matching SaGe v2.0 hyperparameter start scale)
        start = target_vocab_size * 8
        mid = start // 4
        end = target_vocab_size

        # 1. Generate Vocabulary Points (DoubleLinearSchedule, t_mid=0.5, 13 samples)
        n_vocab_samples = 13
        vocabulary_points = []

        for i in range(n_vocab_samples):
            t = i / (n_vocab_samples - 1)
            if t < 0.5:
                # Line 1: Start to Mid (scaled over 0.0 to 0.5)
                val = round((mid - start) * (t / 0.5) + start)
            else:
                # Line 2: Mid to End (scaled over 0.5 to 1.0)
                val = round((end - mid) * ((t - 0.5) / 0.5) + mid)
            vocabulary_points.append(val)

        # 2. Generate Embedding Recomputation Indices (ExponentialDilation a=1.35, 6 samples)
        n_embedding_samples = 6
        a = 1.35
        b = math.log(1 - (1 / a))

        embedding_indices = set()
        for i in range(n_embedding_samples):
            t = i / (n_embedding_samples - 1)
            # Apply exponential dilation to normalized t
            dilated_t = a * (1 - math.exp(b * t))

            # Scale against the length of the vocabulary array (ignoring the very last point)
            idx = round(dilated_t * (n_vocab_samples - 2))
            embedding_indices.add(idx)

        # Map the calculated indices back to the actual vocabulary sizes
        recompute_embeddings_at = [vocabulary_points[j] for j in sorted(embedding_indices)]

        return vocabulary_points, recompute_embeddings_at

    def _get_final_vocab_path(self):
        return f"./models/results/{self.tokenizer_name}/sage_vocabs/active_vocab_{self.vocab_size}.vocab"

    def get_algo_name(self):
        return self.algo_name

    def get_training_corpus_dir(self):
        return self.training_corpus_dir

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        # Decode the raw bytes into a readable UTF-8 string format to avoid HuggingFace encoding byte-strings
        vocab = [byte_tok.decode("utf-8", errors="replace") for byte_tok in self.tokenizer.inv_byte_vocab.values()]
        return vocab