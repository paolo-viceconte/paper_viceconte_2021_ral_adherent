import torch
import argparse
import numpy as np
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
epochs = 150
Te = 10
Tmult = 2
learning_rate_ini = 0.0001
weightDecay_ini = 0.0025
Te_cumulative = Te

# Configure the datasets for training and testing
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

# Normalize weight decay # TODO: check
total_batches = int(len(train_dataloader))
weightDecay_ini = weightDecay_ini / (np.power(total_batches * Te, 0.5))

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

# Initialize learning rate and weight decay schedulers
fake_lr_optimizer = torch.optim.AdamW(mann.parameters(), lr=learning_rate_ini)
fake_wd_optimizer = torch.optim.AdamW(mann.parameters(), lr=weightDecay_ini)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_lr_optimizer, T_max=Te)
wd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fake_wd_optimizer, T_max=Te)

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

writer.close()

print("Training over!")
