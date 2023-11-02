import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation(version):
    """
    Load the evaluation results from the specified version.

    Parameters:
    version (str): The version of the model to load the evaluation results for.

    Returns:
    Any: The evaluation results loaded from the specified version.
    """
    return load_pkl_file(f'outputs/{version}/evaluation.pkl')
