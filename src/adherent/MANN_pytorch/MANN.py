import torch
from torch import nn
from adherent.MANN_pytorch.GatingNetwork import GatingNetwork
from adherent.MANN_pytorch.MotionPredictionNetwork import MotionPredictionNetwork


class MANN(nn.Module):
    """Class for the Mode-Adaptive Neural Network."""

    def __init__(self, train_dataloader, test_dataloader, num_experts, gn_hidden_size, mpn_hidden_size, dropout_probability):
        """Mode-Adaptive Neural Network constructor."""

        # Superclass constructor
        super(MANN, self).__init__()

        # Retrieve input and output dimensions from the training dataset
        train_features, train_labels = next(iter(train_dataloader))
        input_size = train_features.size()[-1]
        output_size = train_labels.size()[-1]

        # Define the two subnetworks composing the MANN architecture
        self.gn = GatingNetwork(input_size=input_size,
                                output_size=num_experts,
                                hidden_size=gn_hidden_size,
                                dropout_probability=dropout_probability)
        self.mpn = MotionPredictionNetwork(num_experts=num_experts,
                                           input_size=input_size,
                                           output_size=output_size,
                                           hidden_size=mpn_hidden_size,
                                           dropout_probability=dropout_probability)

        # Store the dataloaders for training and testing
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def forward(self, x):
        """Mode-Adaptive Neural Network architecture."""

        # Retrieve the output of the Gating Network
        blending_coefficients = self.gn(x.T)

        # Retrieve the output of the Motion Prediction Network
        y = self.mpn(x, blending_coefficients=blending_coefficients)

        return y

    def train_loop(self, loss_fn, optimizer, epoch, writer):
        """Run one epoch of training."""

        # Total number of batches
        total_batches = int(len(self.train_dataloader))

        # Cumulative loss
        cumulative_loss = 0

        # Print the learning rate and weight decay of the current epoch
        print('Current lr:', optimizer.param_groups[0]['lr'])
        print('Current wd:', optimizer.param_groups[0]['weight_decay'])

        # Iterate over batches
        for batch, (X, y) in enumerate(self.train_dataloader):

            # Compute prediction and loss
            pred = self(X.float()).double()
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update cumulative loss
            cumulative_loss += loss.item()

            # Periodically print the current average loss
            if batch % 1000 == 0:
                current_avg_loss = cumulative_loss/(batch+1)
                print(f"avg loss: {current_avg_loss:>7f}  [{batch:>5d}/{total_batches:>5d}]")

        # Print the average loss of the current epoch
        avg_loss = cumulative_loss/total_batches
        print("Final avg loss:", avg_loss)

        # Store the average loss, learning rate and weight decay of the current epoch
        writer.add_scalar('avg_loss', avg_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('wd', optimizer.param_groups[0]['weight_decay'], epoch)
        writer.flush()

    def test_loop(self, loss_fn):
        """Test the trained model on the test data."""

        # Dataset dimension
        num_batches = len(self.test_dataloader)

        # Cumulative loss
        cumulative_test_loss = 0

        with torch.no_grad():

            # Iterate over the testing dataset
            for X, y in self.test_dataloader:
                pred = self(X.float()).double()
                cumulative_test_loss += loss_fn(pred, y).item()

        # Print the average test loss at the current epoch
        avg_test_loss = cumulative_test_loss/num_batches
        print(f"Avg test loss: {avg_test_loss:>8f} \n")

    def inference(self, x):
        """Inference step on the given input x."""

        with torch.no_grad():
            pred = self(x.float()).double()

        return pred

