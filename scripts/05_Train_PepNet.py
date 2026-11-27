'This script trains a convolutional neural network on the PepNet dataset and saves the best model to a pth file in results/Models.'

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from torchvision import transforms
import h5py

device = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU available: ", torch.cuda.is_available())

np.random.seed(69)

# Define functions 
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        """Initializes the MyDataset instance.

        Args:
            data (torch.Tensor): The input data.
            target (torch.Tensor): The target labels corresponding to the input data.
            transform (callable, optional): A function/transform to apply to the input data.
            target_transform (callable, optional): A function/transform to apply to the target labels.
        """
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """Fetches the data and target at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: (x, y) where `x` is the transformed data and `y` is the transformed target.
        """
        x = self.data[index]
        if self.transform:
            x = self.transform(x)

        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)
        
        return x, y
    

# Define the training function
def train(dataloader, model, loss_fn, optimizer):
    """Trains the model for one epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader providing batches of training data.
        model (torch.nn.Module): The neural network model to train.
        loss_fn (callable): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.

    Returns:
        float: The average loss over all batches.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_loss = 0
    model.train()  # Set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)  # Pass the data to the model to execute the model forward
        pred = pred.squeeze(1)  # match target shape [batch] (otherwise MSE broadcasts to [batch, batch])
        loss = loss_fn(pred, y)
        running_loss += loss.item()

        # Backpropagation
        loss.backward()  # Compute gradients of the loss w.r.t parameters (backward pass)
        optimizer.step()  # Do a gradient descent step and adjust parameters
        optimizer.zero_grad()  # Reset the gradients of model parameters to zero (gradients by default add up)

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    running_loss /= num_batches
    return running_loss

# Define the test function
def test(dataloader, model, loss_fn, best_loss, params):
    """Evaluates the model on the validation/test dataset and saves the model if the test loss improves.

    Args:
        dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader providing batches of test data.
        model (torch.nn.Module): The neural network model to evaluate.
        loss_fn (callable): The loss function used to compute the loss.
        best_loss (float): The best loss observed so far; used to determine if the model should be saved.

    Returns:
        tuple: (test_loss, best_loss, best_params) where `test_loss` is the average loss over all batches, `best_loss` is the updated best loss and best_params are the updated best parameters.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    best_params = None
    with torch.no_grad():  # Do not track gradients while evaluating (faster)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Forward pass
            pred = model(X)
            pred = pred.squeeze(1)
            # Compute loss
            batch_loss = loss_fn(pred, y)
            test_loss += batch_loss.item()
            for pred_i, y_i in zip(pred, y):
                if torch.abs(pred_i - y_i) < 0.15 * torch.abs(y_i):
                    correct += 1

    # Average loss over all batches
    test_loss /= num_batches
    correct /= size

    # Save the model if the test loss is the lowest
    if test_loss <= best_loss:
        best_loss = test_loss
        best_params = params
        torch.save(model.state_dict(), '../results/Models/PepNet.pth') 
        print(f'Saving model with validation loss {best_loss:.4f}')
        

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, best_loss, best_params


if __name__ == "__main__":

    # Load the pre-split train/val/test data (see 04_Data_Augmentor.py: the split
    # happens before augmentation so a peptide's augmented copy can never end up
    # on the opposite side of the split from its original image).
    def load_h5(path):
        """Loads an HDF5 split into normalized (float32, NCHW) tensors."""
        with h5py.File(path, 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['permeability'])
        data = torch.from_numpy(images).type(torch.float32).permute(0, 3, 1, 2)
        labels = torch.from_numpy(labels).type(torch.float32)
        return data, labels

    train_data, train_labels = load_h5('../data/PepNet_train_augmented.h5')
    val_data, val_labels = load_h5('../data/PepNet_val.h5')
    test_data, test_labels = load_h5('../data/PepNet_test.h5')

    print(train_data.shape, train_labels.shape)

    # Select a subset of the *training* images for computing mean and std
    subset_images = train_data[:1000]

    # Compute mean and std from the training subset only (never from val/test)
    mean = torch.mean(subset_images.float(), dim=(0, 2, 3))
    std = torch.std(subset_images.float(), dim=(0, 2, 3))

    print("Computed Mean:", mean)
    print("Computed Std:", std)

    # Define the composed transform
    composed_transform = transforms.Compose([
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # Create the datasets
    Train_dataset = MyDataset(train_data, train_labels, transform=composed_transform)
    Val_dataset = MyDataset(val_data, val_labels, transform=composed_transform)
    Test_dataset = MyDataset(test_data, test_labels, transform=composed_transform)


    # Define hyperparameters for the tuning 
    hyperparameters = [
        
        {'epochs': 30, 'batch_size': 32, 'learning_rate': 1e-4},
        {'epochs': 30, 'batch_size': 32, 'learning_rate': 1e-3},
        {'epochs': 30, 'batch_size': 64, 'learning_rate': 1e-3},
        {'epochs': 30, 'batch_size': 64, 'learning_rate': 1e-4},

        {'epochs': 50, 'batch_size': 32, 'learning_rate': 1e-4},
        {'epochs': 50, 'batch_size': 32, 'learning_rate': 1e-3},
        {'epochs': 50, 'batch_size': 64, 'learning_rate': 1e-3},
        {'epochs': 50, 'batch_size': 64, 'learning_rate': 1e-4},

        {'epochs': 80, 'batch_size': 32, 'learning_rate': 1e-4},
        {'epochs': 80, 'batch_size': 32, 'learning_rate': 1e-3},
        {'epochs': 80, 'batch_size': 64, 'learning_rate': 1e-3},
        {'epochs': 80, 'batch_size': 64, 'learning_rate': 1e-4},

        {'epochs': 100, 'batch_size': 32, 'learning_rate': 1e-3},
    ]

    # Define the model architecture as a factory, so each hyperparameter config
    # below trains its own freshly-initialized model. Building a single
    # `cnn_model` here and reusing it across configs would make this loop
    # continue training one model across configs instead of comparing them.
    def build_model():
        return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Flatten(),

        nn.Linear(256*18*18, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(256, 84),
        nn.BatchNorm1d(84),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(84, 1)    # Single output for regression
        ).to(device)


    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    best_params = None
    results_grid_search = []
    for params in hyperparameters:

        nb_epochs = params['epochs']
        bs = params['batch_size']
        lr = params['learning_rate']

        cnn_model = build_model()

        print(f"Training with params: epochs={nb_epochs}, batch_size={bs}, learning_rate={lr}")

        optimizer_full_cnn = torch.optim.Adam(cnn_model.parameters(), lr) # Pass model parameters to optimizer
        
        train_dataloader = DataLoader(Train_dataset, batch_size=bs, shuffle=True)
        val_dataloader = DataLoader(Val_dataset, batch_size=bs)

        # Train the model
        Losses_train_convo = []
        Losses_val_convo = []

        for t in range(nb_epochs):
            print(f"Epoch {t+1}\n")

            train_loss = train(train_dataloader, cnn_model, loss_fn, optimizer_full_cnn)
            Losses_train_convo.append(train_loss)
            print(f"Training Error: \n Avg loss: {train_loss:>8f} \n")
            # `test()` only returns non-None params when this epoch improved on
            # best_loss; only overwrite best_params in that case, otherwise a
            # later non-improving epoch clobbers it back to None.
            val_loss, best_loss, candidate_params = test(val_dataloader, cnn_model, loss_fn, best_loss, params)
            Losses_val_convo.append(val_loss)
            if candidate_params is not None:
                best_params = candidate_params

        results_grid_search.append({
            'params': params,
            'best_loss': best_loss
        })

        # Plot the train and test losses
        plt.figure()
        plt.plot(range(nb_epochs-1),Losses_train_convo[1:])
        plt.plot(range(nb_epochs-1),Losses_val_convo[1:])
        plt.legend(["Train Loss","Validation Loss"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        #plt.ylim(0, 1)
        plt.title("PepNet: Train and Validation Loss as a function of epochs")
        plt.savefig(f"../results/plots/PepNet_losses_e{nb_epochs}_b{bs}_lr{lr}.png")

    print(" Done ! \n Grid search results: ", results_grid_search)
    print("Best hyperparameters: ", best_params)
    
    
    # Load the best model (selected by validation loss)
    best_model = cnn_model
    best_model.load_state_dict(torch.load('../results/Models/PepNet.pth'))

    # Final evaluation on the held-out test set, never used for model/hyperparameter selection
    test_dataloader = DataLoader(Test_dataset, batch_size=best_params['batch_size'])

    best_model.eval()
    true_list = []
    pred_list = []
    total_correct = 0
    total_samples = 0
    mse =0
    threshold = 0.15
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = best_model(images)
            outputs = outputs.squeeze(1)
            true_list.extend(labels.tolist())
            pred_list.extend(outputs.tolist())

            #Compute the mean squared error
            batch_mse = torch.mean((outputs - labels)**2).item()
            mse += batch_mse

            # Compute the number of correct predictions within a threshold of 20% of the true value
            for output, label in zip(outputs, labels):
                if torch.abs(output - label) < threshold * torch.abs(label):
                    total_correct += 1
            total_samples += labels.size(0)

    # Compute the mean squared error
    mse /= len(test_dataloader)

    # Compute the accuracy
    accuracy = total_correct / total_samples * 100
    print(f"Test accuracy: {accuracy:.2f}%")

    # Save the predictions to a csv file
    df = pd.DataFrame({"True Values": true_list, "Predicted Values": pred_list})
    df.to_csv("../results/predictions_PepNet.csv", index=False)

    # Save the accuracy and mse to a text file
    with open("../results/metrics_PepNet.txt", "w") as f:
        f.write(f"Test accuracy: {accuracy:.2f}%\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n")
    
    print("Predictions saved to results/predictions_PepNet.csv", "Metrics saved to results/metrics_PepNet.txt")
