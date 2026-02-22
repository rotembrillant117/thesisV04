from src.stats.basic_stats import tokenization_cases, plot_tokenization_cases, do_basic_stats
from src.stats.cue_stats import do_cue_stats
from .stats_utils import get_categories
from src.stats.compare_stats import do_compare_stats
from src.utils.training_data_utils import inject_cues


def run_basic_stats(all_trials, vocab_size):
    for lang, trial_list in all_trials.items():
        for cur_trial in trial_list:
            algo = cur_trial.get_algo_name()
            ff_data = cur_trial.get_ff()
            injected_ff = inject_cues(ff_data, cur_trial.get_l2())

            categories = get_categories(cur_trial)
            # Pass only the raw False Friends to Tokenization Cases instead of full dictionaries
            base_ff = [data["cued"] for data in injected_ff.values()]

            tok_cases = tokenization_cases(cur_trial.get_base_tokenizer(), base_ff, "en", cur_trial.get_l2(),
                                           categories)
            plot_tokenization_cases(tok_cases, algo, "en", lang, categories, "ff", cur_trial.get_graphs_dir())

            do_basic_stats(cur_trial, vocab_size, injected_ff=injected_ff)
            do_cue_stats(cur_trial, vocab_size)


def run_compare_stats(all_trials, vocab_size):
    pairs = [("BPE", "BPE_SAGE"), ("UNI", "UNI_SAGE")]
    target_category = "same_splits"  # The ideal state we want to check movement towards/from

    for lang, trial_list in all_trials.items():
        # Convert list to dict mapping algo_name -> Trial
        algos = {trial.get_algo_name(): trial for trial in trial_list}

        for base_name, sage_name in pairs:
            # Skip if either base or SAGE trial is missing
            if base_name in algos and sage_name in algos:
                base_trial = algos[base_name]
                sage_trial = algos[sage_name]

                do_compare_stats(base_trial, sage_trial, lang, vocab_size, target_category)


