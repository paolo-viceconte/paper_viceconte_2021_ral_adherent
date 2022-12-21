########################################################################################################################
# CHECK INSTALLATION
########################################################################################################################

# import torch
# x = torch.rand(5, 3)
# print(x)





########################################################################################################################
# 1 - TENSORS
########################################################################################################################

# # Arrays for GPUs!
#
# import torch
# import numpy as np
#
# ######################################################################################## INITIALIZATION
#
# # Initialize from data
# data = [[1, 2],[3, 4]]
# x_data = torch.tensor(data)
#
# # Initialize from np array
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
#
# # Initialize from another tensor while retaining the properties of the original tensor
# x_ones = torch.ones_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")
#
# # Initialize from another tensor while overriding the properties of the original tensor
# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")
#
# # Initialize with random or constant values
# shape = (2,3,)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")
#
# ######################################################################################## ATTRIBUTES
#
# tensor = torch.rand(3,4)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")
#
# # Explicitly move to GPU if available # TODO
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda')
#   print(f"Device tensor has been moved on: {tensor.device}")
#
# ######################################################################################## OPERATIONS
#
# # Indexing and slicing
# tensor = torch.rand(4, 4)
# print('First row: ',tensor[0])
# print('First column: ', tensor[:, 0])
# print('Last column:', tensor[..., -1])
# tensor[:,1] = 0
# print(tensor)
#
# # Joining
# t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)
#
# # Matrix multiplication
# # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# y1 = tensor @ tensor.T
# y2 = tensor.matmul(tensor.T)
# y3 = torch.rand_like(tensor)
# torch.matmul(tensor, tensor.T, out=y3)
#
# #Element-wise product
# # This computes the element-wise product. z1, z2, z3 will have the same value
# z1 = tensor * tensor
# z2 = tensor.mul(tensor)
# z3 = torch.rand_like(tensor)
# torch.mul(tensor, tensor, out=z3)
#
# # Aggregate in a single element + conversion into a numerical value by using item()
# agg = tensor.sum()
# agg_item = agg.item()
# print(agg_item, type(agg_item))
#
# # In-place operations, denoted by _, discouraged
# print(tensor, "\n")
# tensor.add_(5)
# print(tensor)
#
# ######################################################################################## BRIDGE WITH NUMPY
#
# # Tensor to numpy array
# t = torch.ones(5)
# print(f"t: {t}")
# n = t.numpy()
# print(f"n: {n}")
#
# # Changes reflect in the associated numpy array!
# t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")
#
# # Numpy to tensor
# n = np.ones(5)
# print(f"n: {n}")
# t = torch.from_numpy(n)
# print(f"t: {t}")
#
# # Once again, changes reflect in the associated tensor!
# np.add(n, 1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")





# ########################################################################################################################
# # 2 - DATASETS AND DATA LOADERS
# ########################################################################################################################
#
# # we want our dataset code to be decoupled from our model training code for better readability and modularity
# # PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset
# # Dataset stores the samples and their corresponding labels.
# # DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
#
# ######################################################################################## LOADING A DATASET
#
# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
#
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )
#
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
#
#
# ######################################################################################## ITERATING AND VISUALIZING A DATASET
#
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
#
# ######################################################################################## CREATING CUSTOM DATASET
#
# # A custom Dataset class must implement three functions: __init__, __len__, and __getitem__ # TODO
#
# import os
# import pandas as pd
# from torchvision.io import read_image
#
# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
#
# ######################################################################################## PREPARE DATA FOR TRAINING WITH DATALOADERS
#
# # While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to
# # reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval
#
# from torch.utils.data import DataLoader
#
# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
#
# ######################################################################################## ITERATE THROUGH THE DATALOADER
#
# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")





# #######################################################################################################################
# # 3 - TRANSFORMS
# #######################################################################################################################
#
# # Use transforms to perform some manipulation of the data and make it suitable for training
# # transform to modify the features
# # target_transform to modify the labels
#
# import torch
# from torchvision import datasets
#
# # The torchvision.transforms module offers several commonly-used transforms out of the box (i.e. callables containing
# # the transformation logic)
# from torchvision.transforms import ToTensor, Lambda
#
# # ToTensor() converts a PIL image or NumPy ndarray into a FloatTensor and scales the image’s pixel intensity values in the range [0., 1.]
# # Lambda() transforms apply any user-defined lambda function. Here, turn the integer into a one-hot encoded tensor
# ds = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
#     target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
# )




# ########################################################################################################################
# # 4 - BUILD THE NEURAL NETWORK MODEL
# ########################################################################################################################
#
# # Neural networks comprise of layers/modules that perform operations on data.
# # The torch.nn namespace provides all the building blocks you need to build your own neural network.
# # Every module in PyTorch subclasses the nn.Module.
# # A neural network is a module itself that consists of other modules (layers).
# # This nested structure allows for building and managing complex architectures easily.
#
# import os
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
#
# # ######################################################################################## DEFINE THE CLASS
#
# # We define our neural network by subclassing nn.Module,
# # initialize the neural network layers in __init__.
# # Every nn.Module subclass implements the operations on input data in the forward method.
#
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
#
# # Create an instance of NeuralNetwork, and move it to the device, and print its structure.
# model = NeuralNetwork().to(device)
# print(model)
#
# # To use the model, we pass it the input data.
# # This executes the model’s forward, along with some background operations.
# # Do not call model.forward() directly!
#
# # Calling the model on the input returns a 10-dimensional tensor with raw predicted values for each class.
# # We get the prediction probabilities by passing it through an instance of the nn.Softmax module.
#
# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")
#
# # ######################################################################################## MODEL LAYERS
#
# input_image = torch.rand(3,28,28)
# print(input_image.size())
#
# # nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values
# # ( the minibatch dimension (at dim=0) is maintained).
#
# flatten = nn.Flatten()
# flat_image = flatten(input_image)
# print(flat_image.size())
#
# # The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
#
# layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# print(hidden1.size())
#
# # Non-linear activations are what create the complex mappings between the model’s inputs and outputs.
# # They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.
# # In this model, we use nn.ReLU between our linear layers, but there’s other activations to introduce non-linearity in your model.
#
# print(f"Before ReLU: {hidden1}\n\n")
# hidden1 = nn.ReLU()(hidden1)
# print(f"After ReLU: {hidden1}")
#
# # nn.Sequential is an ordered container of modules.
# # The data is passed through all the modules in the same order as defined.
# # You can use sequential containers to put together a quick network like seq_modules.
#
# seq_modules = nn.Sequential(
#     flatten,
#     layer1,
#     nn.ReLU(),
#     nn.Linear(20, 10)
# )
# input_image = torch.rand(3,28,28)
# logits = seq_modules(input_image)
#
# # The last linear layer of the neural network returns logits - raw values in [-infty, infty] - which are passed to the
# # nn.Softmax module. The logits are scaled to values [0, 1] representing the model’s predicted probabilities for each class.
# # dim parameter indicates the dimension along which the values must sum to 1.
#
# softmax = nn.Softmax(dim=1)
# pred_probab = softmax(logits)
#
# # # ######################################################################################## MODEL PARAMETERS
#
# # Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized
# # during training.
# # Subclassing nn.Module automatically tracks all fields defined inside your model object, and makes all parameters # TODO
# # accessible using your model’s parameters() or named_parameters() methods.
# # In this example, we iterate over each parameter, and print its size and a preview of its values.
#
# print(f"Model structure: {model}\n\n")
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")





# # #######################################################################################################################
# # 5- AUTOMATIC DIFFERENTIATION
# # #######################################################################################################################
#
# # When training neural networks, the most frequently used algorithm is back propagation.
# # In this algorithm, parameters (model weights) are adjusted according to the gradient of the loss function with
# # respect to the given parameter.
#
# # You can set the value of requires_grad when creating a tensor, or later by using x.requires_grad_(True) method. # TODO
# # Parameters need the requires_grad!
#
# import torch
#
# x = torch.ones(5)  # input tensor
# y = torch.zeros(3)  # expected output
# w = torch.randn(5, 3, requires_grad=True)
# b = torch.randn(3, requires_grad=True)
# z = torch.matmul(x, w)+b
# loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
#
# ######################################################################################## FUNCTIONS AND COMPUTATIONAL GRAPHS
#
# print(f"Gradient function for z = {z.grad_fn}")
# print(f"Gradient function for loss = {loss.grad_fn}")
#
# ######################################################################################## COMPUTING GRADIENTS
#
# # To optimize weights of parameters in the neural network, we need to compute the derivatives of our loss function
# # with respect to parameters
#
# loss.backward()
# print(w.grad)
# print(b.grad)
#
# ######################################################################################## DISABLING GRADIENT TRACKING
#
# # When we have trained the model and just want to apply it to some input data, i.e. we only want to do forward
# # computations through the network, we can stop tracking computations by surrounding our computation code with
# # torch.no_grad() block. This speeds up computation! # TODO
#
# z = torch.matmul(x, w)+b
# print(z.requires_grad)
#
# with torch.no_grad():
#     z = torch.matmul(x, w)+b
# print(z.requires_grad)
#
# # Another way to achieve the same result is to use the detach() method on the tensor:
#
# z = torch.matmul(x, w)+b
# z_det = z.detach()
# print(z_det.requires_grad)
#
# # DAGs are dynamic in PyTorch !!!!!!!




########################################################################################################################
# 6 - OPTIMIZATION, i.e. TRAINING
########################################################################################################################

# # Now that we have a model and data it’s time to train, validate and test our model by optimizing its parameters on our data.
# # Training a model is an iterative process; in each iteration (called an epoch) the model makes a guess about the output,
# # calculates the error in its guess (loss), collects the derivatives of the error with respect to its parameters
# # and optimizes these parameters using gradient descent.
#
# # ######################################################################################## PREREQUISITE CODE
#
# # From previous sessions:
#
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda
#
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )
#
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
#
# train_dataloader = DataLoader(training_data, batch_size=64)
# test_dataloader = DataLoader(test_data, batch_size=64)
#
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
#
# model = NeuralNetwork()
#
# # ######################################################################################## HYPERPARAMETERS
#
# learning_rate = 1e-3
# batch_size = 64
# epochs = 5
#
# # ######################################################################################## OPTIMIZATION LOOP
#
# # Common loss functions include nn.MSELoss (Mean Square Error) for regression tasks, # TODO
# # and nn.NLLLoss (Negative Log Likelihood) for classification.
# # nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
#
# # We pass our model’s output logits to nn.CrossEntropyLoss, which will normalize the logits and compute the prediction error.
#
# # Initialize the loss function
# # loss_fn = nn.CrossEntropyLoss()
#
# # All optimization logic is encapsulated in the optimizer object.
# # Here, we use the SGD optimizer;
# # additionally, there are many different optimizers available in PyTorch such as ADAM and RMSProp, # TODO
# # that work better for different kinds of models and data
#
# # We initialize the optimizer by registering the model’s parameters that need to be trained,
# # and passing in the learning rate hyperparameter.
#
# # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# # Inside the training loop, optimization happens in three steps:
# # 1. Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; # TODO
# #    to prevent double-counting, we explicitly zero them at each iteration.
# # 2. Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
# # 3. Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
#
# # ######################################################################################## FULL IMPLEMENTATION
#
# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# def test_loop(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0
#
#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# epochs = 20
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")



# #######################################################################################################################
# 7 - SAVE AND LOAD MODEL
# #######################################################################################################################
#
# #######################################################################################  SAVING AND LOADING MODEL WEIGHTS - NO
#
# import torch
# import torchvision.models as models
#
# # PyTorch models store the learned parameters in an internal state dictionary, called state_dict.
# # These can be persisted via the torch.save method
#
# model = models.vgg16(pretrained=True)
# torch.save(model.state_dict(), 'model_weights.pth')
#
# # To load model weights, you need to create an instance of the same model first,
# # and then load the parameters using load_state_dict() method.
#
# # be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to
# # evaluation mode. Failing to do this will yield inconsistent inference results.
#
# model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
# model.load_state_dict(torch.load('model_weights.pth'))
# model.eval()
#
# # ########################################################################################  SAVING AND LOADING MODELS WITH SHAPE - YES # TODO
#
# # When loading model weights, we needed to instantiate the model class first, because the class defines the
# # structure of a network. We might want to save the structure of this class together with the model, in which case we
# # can pass model (and not model.state_dict()) to the saving function:
#
# torch.save(model, 'model.pth')
#
# # We can then load the model like this:
#
# model = torch.load('model.pth')

# ############################################################################################# Saving and loading a general checkpoint in PyTorch
#
# # for inference or resuming training
#
# # When saving a general checkpoint, you must save more than just the model’s state_dict.
# # It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are updated as the model trains. # TODO
# # Other items that you may want to save are the epoch you left off on, the latest recorded training loss,
# # external torch.nn.Embedding layers, and more, based on your own algorithm.
#
# # To save multiple checkpoints, you must organize them in a dictionary and use torch.save() to serialize the dictionary. # TODO
# # A common PyTorch convention is to save these checkpoints using the .tar file extension.
#
# # To load the items, first initialize the model and optimizer, then load the dictionary locally using torch.load().
# # From here, you can easily access the saved items by simply querying the dictionary as you would expect.
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# net = Net()
# print(net)
#
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# # Build your dictionary
#
# EPOCH = 5
# PATH = "model.pt"
# LOSS = 0.4
#
# torch.save({
#             'epoch': EPOCH,
#             'model_state_dict': net.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': LOSS,
#             }, PATH)
#
# # First initialize the model and optimizer, then load the dictionary locally.
#
# model = Net()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
#
# model.eval() # inference
# # - or -
# # model.train() # training





########################################################################################################################
# ADDITIONAL TRAINING
########################################################################################################################

# Zeroing out gradients in PyTorch
#
# It is beneficial to zero out gradients when building a neural network.
# This is because by default, gradients are accumulated in buffers (i.e, not overwritten) whenever .backward() is called.
#
# optimizer.zero_grad() # TODO (investigate)