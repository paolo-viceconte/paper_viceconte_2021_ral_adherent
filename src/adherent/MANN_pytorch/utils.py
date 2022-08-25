import os
import glob
import json
import wget
import os.path
import numpy as np
from typing import List, Dict

def get_dataset_portions(dataset: str) -> Dict:
    """Retrieve the portions associated to each dataset."""

    if dataset == "D2":
        portions = {1: "1_forward_normal_step",
                    2: "2_backward_normal_step",
                    3: "3_left_and_right_normal_step",
                    4: "4_diagonal_normal_step",
                    5: "5_mixed_normal_step"}
    elif dataset == "D3":
        portions = {6: "6_forward_small_step",
                    7: "7_backward_small_step",
                    8: "8_left_and_right_small_step",
                    9: "9_diagonal_small_step",
                    10: "10_mixed_small_step",
                    11: "11_mixed_normal_and_small_step"}

    else:
        raise Exception("Dataset portions only defined for datasets D2 and D3.")

    return portions

def get_latest_model_path(models_path: str) -> str:
    """Retrieve the path of the latest saved model."""

    list_of_files = glob.glob(models_path + '*')
    latest_model = max(list_of_files, key=os.path.getctime)
    print("Latest retrieved model:", latest_model)

    return latest_model

def create_path(path: List) -> None:
    """Create a path if it does not exist."""

    for subpath in path:
        if not os.path.exists(subpath):
            os.makedirs(subpath)

def normalize(X: np.array, axis: int) -> np.array:
    """Normalize X along the given axis."""

    # Compute mean and std
    Xmean = X.mean(axis=axis)
    Xstd = X.std(axis=axis)

    # Avoid division by zero
    for elem in range(Xstd.size):
        if (Xstd[elem] == 0):
            Xstd[elem] = 1

    # Normalize
    X = (X - Xmean) / Xstd

    return X

def denormalize(X: np.array, Xmean: np.array, Xstd: np.array) -> np.array:
    """Denormalize X, given its mean and std."""

    # Denormalize
    X = X * Xstd + Xmean

    return X

def store_in_file(data: list, filename: str) -> None:
    """Store data in file as json."""

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def read_from_file(filename: str) -> np.array:
    """Read data as json from file."""

    with open(filename, 'r') as openfile:
        data = json.load(openfile)

    return np.array(data)

