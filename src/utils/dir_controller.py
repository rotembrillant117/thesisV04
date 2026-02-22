from pathlib import Path

STATS_DIR = Path(Path(__file__).resolve().parent.parent.parent) / 'stats_results'
MODELS_DIR = Path(Path(__file__).resolve().parent.parent.parent) / 'models'
TRIALS_DIR = Path(Path(__file__).resolve().parent.parent.parent) / 'trials'

def create_stats_results_directory(data):
    Path(STATS_DIR).mkdir(parents=True, exist_ok=True)
    vocab_size = data['vocab_size']
    algos = data['algos']
    # l1 = data['l1']['language']
    l2 = [data['l2'][i]['language'] for i in range(len(data['l2']))]

    for l in l2:
        for algo in algos:
            Path(STATS_DIR / f"{vocab_size}" / f"{l}" / f"{algo}" / "graphs").mkdir(parents=True, exist_ok=True)
            Path(STATS_DIR / f"{vocab_size}" / f"{l}" / f"{algo}" / "stats").mkdir(parents=True, exist_ok=True)

def create_model_directories(data):
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    l2 = [data['l2'][i]['language'] for i in range(len(data['l2']))]
    algos = data['algos']
    vocab_size = data['vocab_size']
    for algo in algos:
        if "SAGE" in algo:
            Path(MODELS_DIR / f"results" / f"en_{algo}_{vocab_size}").mkdir(parents=True, exist_ok=True)
            base_algo = algo.split("_")[0]
            Path(MODELS_DIR / f"sp" / f"en_{base_algo}_{vocab_size*8}").mkdir(parents=True, exist_ok=True)
        else:
            Path(MODELS_DIR / f"sp" / f"en_{algo}_{vocab_size}").mkdir(parents=True, exist_ok=True)

    for l in l2:
        for algo in algos:
            if "SAGE" in algo:
                Path(MODELS_DIR / f"results" / f"{l}_{algo}_{vocab_size}").mkdir(parents=True, exist_ok=True)
                Path(MODELS_DIR / f"results" / f"en_{l}_{algo}_{vocab_size}").mkdir(parents=True, exist_ok=True)
                Path(MODELS_DIR / f"results" / f"en_{l}_cues_{algo}_{vocab_size}").mkdir(parents=True, exist_ok=True)
                base_algo = algo.split("_")[0]
                Path(MODELS_DIR / f"sp" / f"{l}_{base_algo}_{vocab_size*8}").mkdir(parents=True, exist_ok=True)
                Path(MODELS_DIR / f"sp" / f"en_{l}_{base_algo}_{vocab_size*8}").mkdir(parents=True, exist_ok=True)
                Path(MODELS_DIR / f"sp" / f"en_{l}_cues_{base_algo}_{vocab_size*8}").mkdir(parents=True, exist_ok=True)

            else:
                Path(MODELS_DIR / f"sp" / f"{l}_{algo}_{vocab_size}").mkdir(parents=True, exist_ok=True)
                Path(MODELS_DIR / f"sp" / f"en_{l}_{algo}_{vocab_size}").mkdir(parents=True, exist_ok=True)
                Path(MODELS_DIR / f"sp" / f"en_{l}_cues_{algo}_{vocab_size}").mkdir(parents=True, exist_ok=True)

def create_trials_directory(data):
    Path(TRIALS_DIR).mkdir(parents=True, exist_ok=True)
    l2 = [data['l2'][i]['language'] for i in range(len(data['l2']))]
    vocab_size = data['vocab_size']
    for l in l2:
        Path(TRIALS_DIR / f"{vocab_size}" / f"{l}").mkdir(parents=True, exist_ok=True)


def create_directories(data):
    create_stats_results_directory(data)
    create_model_directories(data)
    create_trials_directory(data)