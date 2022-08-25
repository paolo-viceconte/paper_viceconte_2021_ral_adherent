import os
import json
import torch
import numpy as np
from torch import nn
from datetime import datetime
from typing import List, Dict
from torch.utils.data import Dataset
from adherent.MANN_pytorch.utils import get_dataset_portions, create_path, normalize, store_in_file


class CustomDataset(Dataset):
    """Class for a custom PyTorch Dataset."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_idx = self.X[idx]
        y_idx = self.Y[idx]
        return x_idx, y_idx

    def get_input_size(self):
        return len(self.X[0])

    def get_output_size(self):
        return len(self.Y[0])


class DataHandler():
    """Class for processing the data in order to get the training and testing sets."""

    def __init__(self, datasets: List, mirroring: bool, training_set_percentage: int):

        # Define the paths to the inputs and outputs of the network
        self.input_paths, self.output_paths = self.define_input_and_output_paths(datasets, mirroring)

        # Define and create the path to store the training-related data
        self.savepath = self.define_savepath(datasets, mirroring)
        create_path([self.savepath])

        # Define the training and testing data
        self.training_data, self.testing_data = self.define_training_and_testing_data(training_set_percentage)

    def define_input_and_output_paths(self, datasets: List, mirroring: bool) -> (List, List):
        """Given the datasets and the mirroring flag, retrieve the list of input and output filenames."""

        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Initialize input filenames list
        input_paths = []

        # Fill input filenames list
        for dataset in datasets:

            inputs = get_dataset_portions(dataset)

            for index in inputs.keys():
                input_path = script_directory + "/../../../datasets/IO_features/inputs_subsampled_" + dataset + "/" + inputs[index] + "_X.txt"
                input_paths.append(input_path)

                if mirroring:
                    input_path = script_directory + "/../../../datasets/IO_features/inputs_subsampled_mirrored_" + dataset + "/" + inputs[index] + "_X_MIRRORED.txt"
                    input_paths.append(input_path)

        # Debug
        print("\nInput files:")
        for input_path in input_paths:
            print(input_path)

        # Initialize output filenames list
        output_paths = []

        # Fill output filenames list
        for dataset in datasets:

            outputs = get_dataset_portions(dataset)

            for index in outputs.keys():
                output_path = script_directory + "/../../../datasets/IO_features/outputs_subsampled_" + dataset + "/" + outputs[index] + "_Y.txt"
                output_paths.append(output_path)

                if mirroring:
                    output_path = script_directory + "/../../../datasets/IO_features/outputs_subsampled_mirrored_" + dataset + "/" + outputs[index] + "_Y_MIRRORED.txt"
                    output_paths.append(output_path)

        # Debug
        print("\nOutput files:")
        for output_path in output_paths:
            print(output_path)

        return input_paths, output_paths

    def define_savepath(self, datasets: List, mirroring: bool) -> str:
        """Given the datasets and the mirroring flag, retrieve the storage path."""

        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Set storage folder
        if not mirroring:
            savepath = script_directory + '/../../../datasets/training_subsampled'
        else:
            savepath = script_directory + '/../../../datasets/training_subsampled_mirrored'
        for dataset in datasets:
            savepath += "_" + dataset

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        savepath += "_" + now

        # Debug
        print("\nSavepath:", savepath, "\n")

        return savepath

    def define_training_and_testing_data(self, training_set_percentage: int) -> (CustomDataset, CustomDataset):
        """Given the training percentage, retrieve the training and testing datasets."""

        # Create the path for the I/O-related storage
        create_path([self.savepath + '/normalization'])

        # ===================
        # RETRIEVE INPUT DATA
        # ===================

        # Initialize input vector
        X = []

        # Collect data from all the input paths
        for input_path in self.input_paths:
            with open(input_path, 'r') as openfile:
                current_input = json.load(openfile)
            X.extend(current_input)

        # Debug
        print("X size:", len(X), "x", len(X[0]))

        # Collect input statistics for denormalization
        X = np.asarray(X)
        Xmean, Xstd = X.mean(axis=0), X.std(axis=0)

        # Normalize input
        X_norm = normalize(X, axis=0)

        # Split into training inputs and test inputs
        splitting_index = round(training_set_percentage / 100 * X.shape[0])
        X_train = X_norm[:splitting_index]
        X_test = X_norm[splitting_index + 1:]

        # Debug
        print("X train size:", len(X_train), "x", len(X_train[0]))

        # Store input statistics
        store_in_file(Xmean.tolist(), self.savepath + "/normalization/X_mean.txt")
        store_in_file(Xstd.tolist(), self.savepath + "/normalization/X_std.txt")

        # ====================
        # RETRIEVE OUTPUT DATA
        # ====================

        # Initialize output vector
        Y = []

        # Collect data from all output paths
        for output_path in self.output_paths:
            with open(output_path, 'r') as openfile:
                current_output = json.load(openfile)
            Y.extend(current_output)

        # Debug
        print("Y size:", len(Y), "x", len(Y[0]))

        # Collect output statistics for denormalization
        Y = np.asarray(Y)
        Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

        # Normalize output
        Y_norm = normalize(Y, axis=0)

        # Split into training outputs and test outputs
        Y_train = Y_norm[:splitting_index]
        Y_test = Y_norm[splitting_index + 1:]

        # Debug
        print("Y train size:", len(Y_train), "x", len(Y_train[0]))

        # Store output statistics
        store_in_file(Ymean.tolist(), self.savepath + "/normalization/Y_mean.txt")
        store_in_file(Ystd.tolist(), self.savepath + "/normalization/Y_std.txt")

        # =====================
        # BUILD CUSTOM DATASETS
        # =====================

        training_data = CustomDataset(X_train, Y_train)
        testing_data = CustomDataset(X_test, Y_test)

        return training_data, testing_data

    def get_savepath(self):
        """Getter of the savepath."""

        return self.savepath

    def get_training_data(self):
        """Getter of the training dataset."""

        return self.training_data

    def get_testing_data(self):
        """Getter of the testing dataset."""

        return self.testing_data
