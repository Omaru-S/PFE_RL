import numpy as np
import pandas as pd
import os

from functools import lru_cache         # NEW

@lru_cache(maxsize=None)                # NEW – keeps one copy per filepath
def _cached_npz_matrix(filepath: str) -> np.ndarray:
    """
    Load the array stored under key 'matrix' in an .npz file and
    keep the result in RAM for subsequent calls.
    """
    with np.load(filepath) as data:
        # Return a read-only view; treat it as immutable elsewhere
        arr = data['matrix']
    return arr


def read_communication_matrix(step):
    """
    Load the matrix from the ./data/full_dense directory with filename step_{step}.npz.

    Parameters:
    step (int): The step number corresponding to the matrix file.

    Returns:
    numpy.ndarray: The loaded matrix from the 'matrix' key.
    """
    filename = f"step_{step}.npz"
    filepath = os.path.join("./data/com_matrices", filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Matrix file '{filename}' not found in './data/com_matrices'.")

    return _cached_npz_matrix(filepath)

def read_vision_matrix(step):
    """
    Load the matrix from the ./data/full_dense directory with filename step_{step}.npz.

    Parameters:
    step (int): The step number corresponding to the matrix file.

    Returns:
    numpy.ndarray: The loaded matrix from the 'matrix' key.
    """
    filename = f"step_{step}.npz"
    filepath = os.path.join("./data/vision_matrices", filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Matrix file '{filename}' not found in './data/vision_matrices'.")

    return _cached_npz_matrix(filepath)

def read_capabilities():
    """
    Load the communication capabilities from the CSV file in ./data/full_dense.

    Returns:
    pandas.DataFrame: The DataFrame containing capabilities data.
    """
    filepath = os.path.join("./data", "com_capabilities.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError("CSV file 'com_capabilities.csv' not found in './data'.")

    return pd.read_csv(filepath)

def nrows_communication_matrix(step: int) -> int:
    """Return the number of rows of the communication matrix for *step*.

    Parameters
    ----------
    step : int
        The time‑step index of the matrix file (…/step_{step}.npz).

    Returns
    -------
    int
        Number of rows (which equals the number of agents) in the matrix.
    """
    return read_communication_matrix(step).shape[0]


def nrows_vision_matrix(step: int) -> int:
    """Return the number of rows of the vision matrix for *step*.

    Parameters
    ----------
    step : int
        The time‑step index of the matrix file (…/step_{step}.npz).

    Returns
    -------
    int
        Number of rows (which equals the number of agents) in the matrix.
    """
    return read_vision_matrix(step).shape[0]