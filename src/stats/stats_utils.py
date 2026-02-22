
def get_categories(trial):
    """
    This functon returns the category of a specific experiment
    :param experiment: an Experiment object
    :return: list of tokenization categories
    """
    l1 = "en"
    l2 = trial.l2
    categories = [f"{l1}_t==multi_t", f"{l2}_t==multi_t", f"{l1}_t=={l2}_t", "same_splits", "different_splits"]
    return categories