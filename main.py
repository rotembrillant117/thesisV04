import json
import sys
from src.tokenizers.train_tokenizers import train_trials, load_trials
from src.stats.run_stats import run_basic_stats, run_compare_stats
from src.utils.dir_controller import create_directories


def parse_args(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data



if __name__ == '__main__':
    train, args_path = sys.argv[1:]
    data = parse_args(args_path)
    create_directories(data)
    if train == "True":
        trials = train_trials(data)
    else:
        trials = load_trials(data['vocab_size'])
    run_basic_stats(trials, data['vocab_size'])
    run_compare_stats(trials, data['vocab_size'])

