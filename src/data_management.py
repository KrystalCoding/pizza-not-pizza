import numpy as np
import pandas as pd
import os
import base64
from datetime import datetime
import joblib


def download_dataframe_as_csv(df):
    """
    Convert a DataFrame to a CSV file and provide a download link.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be converted to CSV.

    Returns:
    str: An HTML download link for the CSV file.

    Example:
    df = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']})
    download_link = download_dataframe_as_csv(df)
    """
    datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" '
        f'download="Report {datetime_now}.csv" '
        f'target="_blank">Download Report</a>'
    )
    return href


def load_pkl_file(file_path):
    """
    Load a Pickle (.pkl) file from the given file path.

    Parameters:
    file_path (str): The path to the Pickle file.

    Returns:
    object: The deserialized object from the Pickle file.

    Example:
    data = load_pkl_file("my_data.pkl")
    """
    return joblib.load(filename=file_path)
