import os
import torch
import argparse
import numpy as np
from torch import nn
from typing import List, Dict
from mann_pytorch.MANN import MANN
from torch.utils.data import DataLoader
from mann_pytorch.utils import create_path
from mann_pytorch.DataHandler import DataHandler
from torch.utils.tensorboard import SummaryWriter

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()
parser.add_argument("--deactivate_mirroring", help="Discard features from mirrored mocap data.", action="store_true")
args = parser.parse_args()
mirroring = not args.deactivate_mirroring

# =====================
# DATASET CONFIGURATION
# =====================

# Auxiliary function to retrieve the portions associated to each dataset
def get_dataset_portions(dataset: str) -> Dict:
    """Retrieve the portions associated to each dataset."""

    if dataset == "D2":
        portions = {
            1: "1_forward_normal_step",
            2: "2_backward_normal_step",
            # 3: "3_left_and_right_normal_step",
            # 4: "4_diagonal_normal_step",
            # 5: "5_mixed_normal_step",
        }
    elif dataset == "D3":
        portions = {
            # 6: "6_forward_small_step",
            # 7: "7_backward_small_step",
            8: "8_left_and_right_small_step",
            # 9: "9_diagonal_small_step",
            # 10: "10_mixed_small_step",
            11: "11_mixed_normal_and_small_step",
        }
    else:
        raise Exception("Dataset portions only defined for datasets D2 and D3.")

    return portions

# Auxiliary function to define the input and output filenames
def define_input_and_output_paths(datasets: List, mirroring: bool) -> (List, List):
    """Given the datasets and the mirroring flag, retrieve the list of input and output filenames."""

    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Initialize input filenames list
    input_paths = []

    # Fill input filenames list
    for dataset in datasets:

        inputs = get_dataset_portions(dataset)

        for index in inputs.keys():
            input_path = script_directory + "/../datasets/IO_features/inputs_subsampled_" + dataset + "/" + inputs[index] + "_X.txt"
            input_paths.append(input_path)

            if mirroring:
                input_path = script_directory + "/../datasets/IO_features/inputs_subsampled_mirrored_" + dataset + "/" + inputs[index] + "_X_MIRRORED.txt"
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
            output_path = script_directory + "/../datasets/IO_features/outputs_subsampled_" + dataset + "/" + outputs[index] + "_Y.txt"
            output_paths.append(output_path)

            if mirroring:
                output_path = script_directory + "/../datasets/IO_features/outputs_subsampled_mirrored_" + dataset + "/" + outputs[index] + "_Y_MIRRORED.txt"
                output_paths.append(output_path)

    # Debug
    print("\nOutput files:")
    for output_path in output_paths:
        print(output_path)

    return input_paths, output_paths

# Auxiliary function to define where to store the training results
def define_storage_folder(datasets: List, mirroring: bool) -> str:
    """Given the datasets and the mirroring flag, retrieve the storage folder."""

    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Set storage folder
    if not mirroring:
        storage_folder = script_directory + '/../datasets/training'
    else:
        storage_folder = script_directory + '/../datasets/training_mirrored'
    for dataset in datasets:
        storage_folder += "_" + dataset

    # Debug
    print("\nStorage folder:", storage_folder, "\n")

    return storage_folder

# Define the datasets to be used
datasets = ["D2", "D3"]

# Retrieve inputs and outputs global paths
input_paths, output_paths = define_input_and_output_paths(datasets, mirroring)

# Retrieve global storage folder
storage_folder = define_storage_folder(datasets, mirroring)

# Retrieve the training and testing datasets
data_handler = DataHandler(input_paths=input_paths, output_paths=output_paths, storage_folder=storage_folder,
                           training=True, training_set_percentage=98)
training_data = data_handler.get_training_data()
testing_data = data_handler.get_testing_data()

# ======================
# TRAINING CONFIGURATION
# ======================

# Random seed
torch.manual_seed(23456)

# Training hyperparameters
num_experts = 4
batch_size = 32
dropout_probability = 0.3
gn_hidden_size = 32
mpn_hidden_size = 512
epochs = 150
Te = 10
Tmult = 2
learning_rate_ini = 0.0001
weightDecay_ini = 0.0025
Te_cumulative = Te

# Configure the datasets for training and testing
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

# Normalize weight decay
total_batches = int(len(train_dataloader))
weightDecay_ini = weightDecay_ini / (np.power(total_batches * Te, 0.5))

# Initialize the MANN architecture
mann = MANN(train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_experts=num_experts,
            gn_hidden_size=gn_hidden_size,
            mpn_hidden_size=mpn_hidden_size,
            dropout_probability=dropout_probability)

# Check the trainable parameters in the model
for name, param in mann.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Check whether the gpu or the cpu is used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Debug
input("\nPress Enter to start the training")

# Use unweighted MSE loss
loss_fn = nn.MSELoss(reduction="mean")

# Initialize the optimizer
optimizer = torch.optim.AdamW(mann.parameters(), lr=learning_rate_ini, weight_decay=weightDecay_ini)

# Initialize learning rate and weight decay schedulers
fake_lr_optimizer = torch.optim.AdamW(mann.parameters(), lr=learning_rate_ini)
fake_wd_optimizer = torch.optim.AdamW(mann.parameters(), lr=weightDecay_ini)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_lr_optimizer, T_max=Te)
wd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_wd_optimizer, T_max=Te)

# Configure tensorboard writer
writer_path = data_handler.get_savepath() + "/logs/"
create_path(writer_path)
writer = SummaryWriter(log_dir=writer_path)

# Create the path to periodically store the learned models
model_path = data_handler.get_savepath() + "/models/"
create_path(model_path)
last_model_path = ""

# =============
# TRAINING LOOP
# =============

for epoch in range(epochs):

    # Debug
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # Perform one epoch of training and testing
    mann.train_loop(loss_fn, optimizer, epoch, writer)
    mann.test_loop(loss_fn)

    # Save the trained model periodically and at the very last iteration
    if epoch > 0 and (epoch % 10 == 0 or epoch == epochs - 1):
        current_model_path = model_path + "/model_" + str(epoch) + ".pth"
        torch.save(mann, current_model_path)
        last_model_path = current_model_path

    # Update current learning rate and weight decay
    lr_scheduler.step()
    wd_scheduler.step()
    optimizer.param_groups[0]['lr'] = lr_scheduler.get_last_lr()[0]
    optimizer.param_groups[0]['weight_decay'] = wd_scheduler.get_last_lr()[0]

    # Reinitialize learning rate and weight decay
    if epoch == Te_cumulative - 1:
        Te = Tmult * Te
        Te_cumulative += Te
        fake_lr_optimizer = torch.optim.AdamW(mann.parameters(), lr=learning_rate_ini)
        fake_wd_optimizer = torch.optim.AdamW(mann.parameters(), lr=weightDecay_ini)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_lr_optimizer, T_max=Te)
        wd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_wd_optimizer, T_max=Te)

# Close tensorboard writer
writer.close()

# Debug
print("Training over!")
