# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
import fairseq_cli.preprocess
from omegaconf import II
import sys

import numpy as np
from fairseq import utils
from fairseq.logging import metrics
from fairseq.tasks import translation
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq import options
import fairseq_cli


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def retokenize_and_load_langpair_dataset(
    raw_data_path, # where raw text is held
    data_path, # where tokenized data, bpe data, and binarized data is held
    bpe_impl_path, # the path to the tokenizer impliementation (e.g. "/home/marco/github/fairseq/examples/translation_decoupled_vocab/subword-nmt/subword_nmt")
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    src_dropout=0.0,
    tgt_dropout=0.0,
    # TODO ADD SEED FOR DROPOUT(s)
    # src_dropout_seed = 0,
    # tgt_dropout_seed = 0,
):
    # BPE_DIR = "/home/marco/github/fairseq/examples/translation_decoupled_vocab/subword-nmt/subword_nmt"
    BPE_DIR = bpe_impl_path

    logger.info(f"DATA PATH: {data_path}")

    src_model_path = os.path.join(data_path, f"{src}_tokenizer.model")
    tgt_model_path = os.path.join(data_path, f"{tgt}_tokenizer.model")

    src_vocab_path = os.path.join(data_path, f"{src}_tokenizer.dict")
    tgt_vocab_path = os.path.join(data_path, f"{tgt}_tokenizer.dict")

    src_raw_path = os.path.join(raw_data_path, f"{split}.{src}")
    tgt_raw_path = os.path.join(raw_data_path, f"{split}.{tgt}")

    src_tokenized_path = os.path.join(data_path, f"{split}.{src}")
    tgt_tokenized_path = os.path.join(data_path, f"{split}.{tgt}")

    if src_dropout > 0.0:
        # os.system(
        #     f"python3 {BPE_DIR}/apply_bpe.py --dropout {src_dropout} -c {src_code_path} --vocabulary {src_vocab_path} --vocabulary-threshold {src_threshold} < {src_raw_path} > {src_tokenized_path}"
        # )
        if split == 'train':
            os.system(f"rm {data_path}/train.*.bin")
            os.system(f"rm {data_path}/train.*.idx")
        # os.system(
        #     f"python3 {BPE_DIR}/encode_spm.py --dropout {src_dropout} -c {src_code_path} --vocabulary {src_vocab_path} < {src_raw_path} > {src_tokenized_path}"
        # )
        os.system(f"spm_encode --model={src_model_path} --alpha={src_dropout} --output_format=sample_piece --input={src_raw_path} --output={src_tokenized_path}")
        # spm_encode --model=spm.model --vocabulary={vocab_file}.L2 --vocabulary_threshold=50 < {test_file}.L2 > {test_file}.seg.L2
        
    else:
        # os.system(
        #     f"python3 {BPE_DIR}/apply_bpe.py -c {src_code_path} --vocabulary {src_vocab_path} --vocabulary-threshold {src_threshold} < {src_raw_path} > {src_tokenized_path}"
        # )
        # os.system(
        #     f"python3 {BPE_DIR}/apply_bpe.py -c {src_code_path} --vocabulary {src_vocab_path} < {src_raw_path} > {src_tokenized_path}"
        # )
        os.system(f"spm_encode --model={src_model_path} --output_format=piece --input={src_raw_path} --output={src_tokenized_path}")

    if tgt_dropout > 0.0:
        # os.system(
        #     f"python3 {BPE_DIR}/apply_bpe.py --dropout {tgt_dropout} --vocabulary {tgt_vocab_path} --vocabulary-threshold {tgt_threshold} -c {tgt_code_path} < {tgt_raw_path} > {tgt_tokenized_path}"
        # )
        if split == 'train':
            os.system(f"rm {data_path}/train.*.bin")
            os.system(f"rm {data_path}/train.*.idx")
        # os.system(
        #     f"python3 {BPE_DIR}/apply_bpe.py --dropout {tgt_dropout} --vocabulary {tgt_vocab_path} -c {tgt_code_path} < {tgt_raw_path} > {tgt_tokenized_path}"
        # )
        os.system(f"spm_encode --model={tgt_model_path} --alpha={tgt_dropout} --output_format=sample_piece --input={tgt_raw_path} --output={tgt_tokenized_path}")
    else:
        # os.system(
        #     f"python3 {BPE_DIR}/apply_bpe.py -c {tgt_code_path} --vocabulary {tgt_vocab_path} --vocabulary-threshold {tgt_threshold} < {tgt_raw_path} > {tgt_tokenized_path}"
        # )
        # os.system(
        #     f"python3 {BPE_DIR}/apply_bpe.py -c {tgt_code_path} --vocabulary {tgt_vocab_path} < {tgt_raw_path} > {tgt_tokenized_path}"
        # )
        os.system(f"spm_encode --model={tgt_model_path} --output_format=piece --input={tgt_raw_path} --output={tgt_tokenized_path}")

    with open(src_tokenized_path, 'r') as f:
        logger.info("LOGGING SOME RANDOM TOKENIZED LINES (src)")
        logger.info("====================================")
        for _ in range(10):
            logger.info(f.readline().strip())

    with open(tgt_tokenized_path, 'r') as f:
        logger.info("LOGGING SOME RANDOM TOKENIZED LINES (tgt)")
        logger.info("====================================")
        for _ in range(10):
            logger.info(f.readline().strip())
            

    # TEXT=examples/translation_dropout/experiments/$EXPERIMENT_NAME
    # fairseq-preprocess --source-lang $src --target-lang $tgt \
    #     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    #     --destdir examples/translation_dropout/data-bin/$EXPERIMENT_NAME \
    #     --workers 8 \
    #     --srcdict examples/translation_dropout/experiments/$EXPERIMENT_NAME/vocab.$src \
    #     --tgtdict examples/translation_dropout/experiments/$EXPERIMENT_NAME/vocab.$tgt


    parser = options.get_preprocessing_parser()
    args = parser.parse_args(['--source-lang', src, '--target-lang', tgt, '--srcdict', src_vocab_path, '--tgtdict', tgt_vocab_path, '--trainpref', os.path.join(data_path, 'train'), '--validpref', os.path.join(data_path, 'valid'), '--testpref', os.path.join(data_path, 'test'), '--destdir', data_path, '--workers', '8'])
    logger.info("STARTING PREPROCESS")
    logger.info(args)
    fairseq_cli.preprocess.main(args)
    # os.system(
    #     f"fairseq-preprocess --source-lang {src} --target-lang {tgt} --srcdict {src_vocab_path} --tgtdict {tgt_vocab_path} --trainpref {os.path.join(data_path, 'train')} --destdir {data_path} --workers 20"
    # )
    logger.info("ENDING PREPROCESS")
    # os.system(
    #     f"nohup fairseq-preprocess --source-lang {src} --target-lang {tgt} --srcdict {src_vocab_path} --tgtdict {tgt_vocab_path} --trainpref {os.path.join(data_path, 'train')} --validpref {os.path.join(data_path, 'valid')} --testpref {os.path.join(data_path, 'test')} --destdir {data_path} --workers 20"
    # )

    # os.system(
    #     f"fairseq-preprocess --source-lang {src} --target-lang {tgt} --thresholdsrc {src_threshold} --thresholdtgt {tgt_threshold} --trainpref {os.path.join(data_path, 'train')} --destdir {data_path} --workers 20"
    # )
    logger.info("MARCO: LOAD LANG PAIR DATASET")

    # maybe replace with translation.load_langpair_dataset(......)

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class RegularizationTranslationConfig(translation.TranslationConfig):
    src_dropout: Optional[float] = field(
        default=0.0, metadata={"help": "the source dropout rate, 0.0 if not used"}
    )
    tgt_dropout: Optional[float] = field(
        default=0.0, metadata={"help": "the target dropout rate, 0.0 if not used"}
    )
    raw_data_path: Optional[str] = field(
        default="", metadata={"help": "location of the raw (untokenized) corpus. corpus should be presplit into train/test/valid"}
    )
    bpe_impl_path: Optional[str] = field(
        default="", metadata={"help": "location of the bpe implemetation"}
    )
    jamo_type: Optional[str] = field(default="syllable", metadata={"help": "the character encoding type (syllable, compat, positional)"})

# def retokenize_and_load_langpair_dataset(
#     raw_data_path, # where raw text is held
#     data_path, # where tokenized data, bpe data, and binarized data is held
#     bpe_impl_path, # the path to the tokenizer impliementation (e.g. "/home/marco/github/fairseq/examples/translation_decoupled_vocab/subword-nmt/subword_nmt")
#     split,
#     src,

@register_task(
    "translation-with-subword-regularization", dataclass=RegularizationTranslationConfig
)
class TranslationWithSubwordRegularizationTask(translation.TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: RegularizationTranslationConfig

    def __init__(self, cfg: RegularizationTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    @classmethod
    def setup_task(cls, cfg: RegularizationTranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # if cfg.jamo_type in ["syllable", "compat", "positional", "syllable_small", "compat_small", "positional_small"]:
        #     cfg.eval_bleu_remove_bpe = cfg.jamo_type
        # else:
        #     raise ValueError(
        #         f"Invalid jamo_type {cfg.jamo_type}"
        #     )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        logger.info("MARCO: LOAD DATASET")
        logger.info(f"MARCO: {data_path}")

        # only retokenize the dataset when its the training split and we are using dropout
        if split == "train" and (
            self.cfg.src_dropout > 0.0 or self.cfg.tgt_dropout > 0.0
        ):
            logging.info("MARCO RELOAD AND RETOKENIZE DATASET")
            self.datasets[split] = retokenize_and_load_langpair_dataset(
                self.cfg.raw_data_path,
                data_path,
                self.cfg.bpe_impl_path,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
                src_dropout=self.cfg.src_dropout,
                tgt_dropout=self.cfg.tgt_dropout,
            )
        else:
            logging.info("MARCO LOADING ALREADY MADE DATASET")
            self.datasets[split] = translation.load_langpair_dataset(
                data_path,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
            )

    def has_non_static_dataset(self):
        return self.cfg.src_dropout > 0 or self.cfg.tgt_dropout > 0
