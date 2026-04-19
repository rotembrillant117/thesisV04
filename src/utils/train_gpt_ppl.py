import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GPT_DATA_DIR = PROJECT_ROOT / "gpt_data"

VOCAB_SIZE = 8001      # 0..7999 real sentencepiece ids, 8000 reserved for padding later
PAD_ID = 8000
BLOCK_SIZE = 128
STRIDE = 64

N_EMBD = 256
N_LAYER = 4
N_HEAD = 8

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
SEED = 42


class TokenBlockDataset(Dataset):
    """
    Dataset class that creates token blocks for GPT training and evaluation
    """
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def parse_args():
    """
    Parse command line arguments for gpt training
    :return:
    """
    parser = argparse.ArgumentParser(description="Train GPT and compute perplexity.")
    parser.add_argument("--lang_pair", required=True, help="Example: en_de")
    parser.add_argument("--tokenizer_type", required=True, choices=["BPE", "UNI"])
    parser.add_argument("--condition", required=True, choices=["baseline", "cued"])
    return parser.parse_args()


def load_token_sequences(path):
    """
    Loads the token sequences from a file path of type .npy
    :param path: the file path
    :return:
    """
    arr = np.load(path, allow_pickle=True)
    return [list(seq) for seq in arr]


def flatten_sequences(sequences):
    """
    Flatten the sequences into one list of token id stream
    :param sequences: list of token sequences
    :return: list of all token sequences
    """
    flat = []
    for seq in sequences:
        flat.extend(int(x) for x in seq)
    return flat


def build_blocks(flat_ids, block_size):
    """
    Builds the blocks for gpt input
    :param flat_ids: the token id stream
    :param block_size: the length of the blocks
    :return: list of block ids
    """
    n_full_blocks = len(flat_ids) // block_size
    usable_len = n_full_blocks * block_size
    flat_ids = flat_ids[:usable_len]

    blocks = []
    for i in range(0, usable_len, block_size):
        blocks.append(flat_ids[i:i + block_size])

    return blocks


def get_npy_paths(lang_pair, tokenizer_type, condition):
    """
    Gets the file paths for train and eval data, for the specified language pair, tokenizer type and condition (baseline or cued)
    :param lang_pair: the language pair
    :param tokenizer_type: the tokenizer type
    :param condition: baseline or cued
    :return: file paths
    """
    pair_dir = GPT_DATA_DIR / lang_pair

    if not pair_dir.exists():
        raise ValueError(f"Missing language pair folder: {pair_dir}")

    train_path = pair_dir / f"{condition}_train_{tokenizer_type}_8000_ids.npy"
    eval_path = pair_dir / f"{condition}_eval_{tokenizer_type}_8000_ids.npy"

    if not train_path.exists():
        raise ValueError(f"Missing train file: {train_path}")

    if not eval_path.exists():
        raise ValueError(f"Missing eval file: {eval_path}")

    return train_path, eval_path, pair_dir


def build_model():
    """
    Builds the GPT model
    :return: gpt model
    """
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        pad_token_id=PAD_ID,
    )
    return GPT2LMHeadModel(config)


def compute_perplexity_strided(model, flat_eval_ids, device, max_length, stride):
    """
    Computes the perplexity by sliding window, like advised at:
    https://huggingface.co/docs/transformers/perplexity?utm_source=chatgpt.com
    :param model: the model
    :param flat_eval_ids: the token id stream of eval data
    :param device: the device
    :param max_length: the maximum length of the sequence
    :param stride: the stride of the sliding window
    :return:
    """
    model.eval()

    input_ids_all = torch.tensor(flat_eval_ids, dtype=torch.long, device=device).unsqueeze(0)
    seq_len = input_ids_all.size(1)

    nll_sum = torch.tensor(0.0, device=device)
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = input_ids_all[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size

        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)

    return avg_nll.item(), ppl.item()


def main():
    args = parse_args()

    train_path, eval_path, pair_dir = get_npy_paths(
        lang_pair=args.lang_pair,
        tokenizer_type=args.tokenizer_type,
        condition=args.condition,
    )

    train_sequences = load_token_sequences(train_path)
    eval_sequences = load_token_sequences(eval_path)

    flat_train_ids = flatten_sequences(train_sequences)
    flat_eval_ids = flatten_sequences(eval_sequences)

    train_blocks = build_blocks(flat_train_ids, BLOCK_SIZE)
    eval_blocks = build_blocks(flat_eval_ids, BLOCK_SIZE)

    if len(train_blocks) == 0:
        raise ValueError("No train blocks were created. The training token stream is too short.")

    if len(eval_blocks) == 0:
        raise ValueError("No eval blocks were created. The eval token stream is too short.")

    train_dataset = TokenBlockDataset(train_blocks)
    eval_dataset = TokenBlockDataset(eval_blocks)

    model = build_model()

    output_dir = pair_dir / f"gpt_{args.condition}_{args.tokenizer_type.lower()}"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    eval_results = trainer.evaluate()

    device = model.device
    avg_nll, ppl = compute_perplexity_strided(
        model=model,
        flat_eval_ids=flat_eval_ids,
        device=device,
        max_length=BLOCK_SIZE,
        stride=STRIDE,
    )

    results = {
        "lang_pair": args.lang_pair,
        "tokenizer_type": args.tokenizer_type,
        "condition": args.condition,
        "train_file": train_path.name,
        "eval_file": eval_path.name,
        "block_size": BLOCK_SIZE,
        "stride": STRIDE,
        "vocab_size": VOCAB_SIZE,
        "pad_id": PAD_ID,
        "n_embd": N_EMBD,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "num_epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "num_train_sequences": len(train_sequences),
        "num_eval_sequences": len(eval_sequences),
        "num_train_tokens": len(flat_train_ids),
        "num_eval_tokens": len(flat_eval_ids),
        "num_train_blocks": len(train_blocks),
        "num_eval_blocks": len(eval_blocks),
        "trainer_eval_loss": float(eval_results["eval_loss"]),
        "strided_eval_nll": avg_nll,
        "perplexity": ppl,
    }

    results_path = pair_dir / f"gpt_results_{args.condition}_{args.tokenizer_type}_8000.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log_history_path = pair_dir / f"gpt_log_history_{args.condition}_{args.tokenizer_type}_8000.json"
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    print(f"Trainer eval loss: {eval_results['eval_loss']:.6f}")
    print(f"Strided eval NLL: {avg_nll:.6f}")
    print(f"Perplexity: {ppl:.6f}")
    print(f"Saved results to: {results_path}")
    print(f"Saved log history to: {log_history_path}")


if __name__ == "__main__":
    main()