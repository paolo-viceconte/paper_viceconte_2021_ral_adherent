import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from adherent.MANN_pytorch.MANN import MANN
from torch.utils.tensorboard import SummaryWriter
from adherent.MANN_pytorch.DataHandler import DataHandler
from adherent.MANN_pytorch.utils import create_path

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

# Define the datasets to be used
datasets = ["D2", "D3"]

# Retrieve the training and testing datasets
data_handler = DataHandler(datasets=datasets, mirroring=mirroring, training_set_percentage=98)
training_data = data_handler.get_training_data()
testing_data = data_handler.get_testing_data()

# ========
# TRAINING
# ========

# Debug
input("\nPress Enter to start the training")

# Random seed
torch.manual_seed(23456)

# Define training hyperparameters
num_experts = 4
batch_size = 32
dropout_probability = 0.3
gn_hidden_size = 32
mpn_hidden_size = 512
epochs = 100
# TODO: are lr and wd ok even if we do not reset them?
learning_rate_ini = 0.0001
weightDecay_ini = 0.0025
# TODO: additional parameters to reset the lr and wd, not used for the time being
# Te = 10
# Tmult = 2

# Configure the datasets for training and testing
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

# Initialize the MANN architecture
mann = MANN(train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            num_experts=num_experts,
            gn_hidden_size=gn_hidden_size,
            mpn_hidden_size=mpn_hidden_size,
            dropout_probability = dropout_probability)

# Check the trainable parameters in the model
for name, param in mann.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Check whether the gpu or the cpu is used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define the loss function
loss_fn = nn.MSELoss(reduction="mean")

# Initialize the optimizer
optimizer = torch.optim.AdamW(mann.parameters(), lr=learning_rate_ini, weight_decay=weightDecay_ini)

# Configure tensorboard writer
writer_path = data_handler.get_savepath() + "/logs/"
create_path([writer_path])
writer = SummaryWriter(log_dir=writer_path)

# Create the path to periodically store the learned models
model_path = data_handler.get_savepath() + "/models/"
create_path([model_path])
last_model_path = ""

# Training loop
for epoch in range(epochs):

    print(f"Epoch {epoch + 1}\n-------------------------------")
    mann.train_loop(loss_fn, optimizer, epoch, writer)
    mann.test_loop(loss_fn, epoch, writer)

    # Save the trained model periodically and at the very last iteration
    if epoch % 10 == 0 or epoch == epochs - 1:
        current_model_path = model_path + "/model_" + str(epoch) + ".pth"
        torch.save(mann, current_model_path)
        last_model_path = current_model_path

writer.close()

print("Training over!")
