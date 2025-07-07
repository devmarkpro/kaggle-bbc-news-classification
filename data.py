import pandas as pd


def load_data(train_path: str, test_path: str, sample_solution_path: str) -> tuple:
    """
    Load the train, test, and sample solution datasets.

    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.
        sample_solution_path (str): Path to the sample solution dataset.

    Returns:
        tuple: A tuple containing the train, test, and sample solution DataFrames.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_solution_df = pd.read_csv(sample_solution_path)

    return train_df, test_df, sample_solution_df
