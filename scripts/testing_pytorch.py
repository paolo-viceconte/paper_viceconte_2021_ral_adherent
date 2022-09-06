import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from adherent.MANN_pytorch.DataHandler import DataHandler

# =============
# CONFIGURATION
# =============

# Path to the learned model
script_directory = os.path.dirname(os.path.abspath(__file__))
last_model_path = script_directory + "/storage_20220906-155556/models/model_9.pth"

# Local input, output and storage paths
input_paths = ["input_1.txt"]
output_paths = ["output_1.txt"]
storage_folder = "storage"

# Global input, output and storage paths
for i in range(len(input_paths)):
    input_paths[i] = script_directory + "/" + input_paths[i]
for i in range(len(output_paths)):
    output_paths[i] = script_directory + "/" + output_paths[i]
storage_folder = script_directory + "/" + storage_folder

# Retrieve the testing dataset, iterable on batches of one single element
data_handler = DataHandler(input_paths=input_paths, output_paths=output_paths, storage_folder=storage_folder, training_set_percentage=98)
testing_data = data_handler.get_testing_data()
batch_size = 1
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

# Define the loss function
loss_fn = nn.MSELoss(reduction="mean")

# ===============
# MODEL RESTORING
# ===============

# Restore the model with the trained weights
mann_restored = torch.load(last_model_path)

# Set dropout and batch normalization layers to evaluation mode before running inference
mann_restored.eval()

# ============
# TESTING LOOP
# ============

# Perform one testing
mann_restored.test_loop(loss_fn)

# ==============
# INFERENCE LOOP
# ==============

# Perform inference on each element of the test set
for X, y in test_dataloader:

    # Inference
    pred = mann_restored.inference(X)

    # Debug
    print()
    print("##################################################################################")
    print("INPUT:")
    print(X)
    print("GROUND TRUTH:")
    print(y)
    print("OUTPUT:")
    print(pred)
    print()
    input("Press ENTER to continue inference")


