"This is a script to fine-tune Google's Vision Transformer model on the PepNet dataset for permeability prediction."

# Importing core scientific libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: plt.show() would otherwise block
                       # waiting for a GUI window to be closed when a display is present,
                       # which hangs an unattended run indefinitely.
import matplotlib.pyplot as plt

# Importing image processing libraries
from PIL import Image, ExifTags  

# Importing libraries for working with HDF5 files
import h5py

# Importing PyTorch libraries for deep learning
import torch  
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader  

# Importing Hugging Face datasets library
from datasets import load_dataset  
import datasets  
from datasets import Dataset  

# Importing Hugging Face transformers library for Vision Transformer
from transformers import ViTImageProcessor  
from transformers import TrainingArguments, Trainer  
from transformers import ViTForImageClassification

# Importing scikit-learn libraries
from sklearn.metrics import mean_squared_error



def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return dict(mean_squared_error=mean_squared_error(predictions, labels))
    

if __name__ == "__main__":
    
    ######################################### Load the data #########################################

    # Load the pre-split train/val/test data (see 04_Data_Augmentor.py: the split
    # happens before augmentation so a peptide's augmented copy can never end up
    # on the opposite side of the split from its original image).
    def load_h5_as_images(path):
        with h5py.File(path, 'r') as F:
            images = np.array(F['images'])
            images = np.clip(images, 0, 255).astype(np.uint8)
            labels = np.array(F['permeability'])
        return [Image.fromarray(img) for img in images], labels

    train_data, train_labels = load_h5_as_images('../data/PepNet_train_augmented.h5')
    val_data, val_labels = load_h5_as_images('../data/PepNet_val.h5')
    test_data, test_labels = load_h5_as_images('../data/PepNet_test.h5')

    #Verify the data
    print(train_labels[0])
    plt.imshow(train_data[0])
    plt.show()
    print(train_data[0], train_labels.shape)

    train_dict = {"img": train_data, "label": train_labels}
    val_dict = {"img": val_data, "label": val_labels}
    test_dict = {"img": test_data, "label": test_labels}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    ######################################### Preprocessing the data #########################################

    # Load the ViT image processor
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Define the transforms
    from torchvision.transforms import (CenterCrop, 
                                        Compose, 
                                        Normalize, 
                                        RandomHorizontalFlip,
                                        RandomResizedCrop, 
                                        Resize, 
                                        ToTensor)

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    # Apply the transforms
    train_dataset.set_transform(train_transforms)
    val_dataset.set_transform(val_transforms)
    test_dataset.set_transform(val_transforms)


    ######################################### Define the model #########################################

    class ViTForRegression(ViTForImageClassification):
        def __init__(self, config):
            super().__init__(config)
            # Add additional layers before the final output layer
            self.additional_layers = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),  # Add a linear layer
                nn.ReLU(),  # Add activation function
                nn.Linear(config.hidden_size, config.hidden_size)  # Add another linear layer
            )
            # Modify the output layer for regression
            self.classifier = nn.Linear(config.hidden_size, 1)  # Output a single value

    
    # Create model instance
    model = ViTForRegression.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=1) #num_labels=1 to solve the target dimension mismatch with the input
    model.base_model.config
    # Newer transformers versions pick the loss function by matching the class
    # name against known head types (e.g. "ForImageClassification"); our
    # renamed "ViTForRegression" subclass doesn't match anything and silently
    # falls back to a causal-LM loss. Set it explicitly so num_labels=1 still
    # dispatches to the regression MSELoss path.
    model.loss_type = "ForImageClassification"
    model.config.problem_type = "regression"

    metric_name = "mean_squared_error"

    args = TrainingArguments(
        f"../results/Models/ViT_Tuned",
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,  # only keep the best + latest checkpoint (100 epochs * ~1GB each otherwise)
        learning_rate=5e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=100,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        remove_unused_columns=False,
    )

    # Create a Trainer instance
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    ######################################### Train the model #############################################
    trainer.train()

    ###################################### Evaluate the model #############################################
    outputs = trainer.predict(test_dataset)
    
    # Save the metrics
    with open("../results/ViT_Tuned_Metrics.txt", "w") as f:
        f.write(str(outputs.metrics))

    # Save the predictions (the regression head outputs shape (N, 1); flatten to (N,))
    df= pd.DataFrame({"True Values": outputs.label_ids, "Predicted Values": outputs.predictions.reshape(-1)})
    df.to_csv("../results/ViT_Tuned_Predictions.csv", index=False)

    ######################################### Plot the predictions and true values #########################

    # Load the predictions from the CSV file
    df = pd.read_csv("../results/ViT_Tuned_Predictions.csv")

    # Extract the predicted and true values
    predicted_values = df["Predicted Values"]
    true_values = df["True Values"]
    index = [i for i in range(len(true_values))]

    # Plot the predicted and true values
    plt.figure()
    plt.scatter(index,true_values, label="True Values", s=10)
    plt.scatter(index,predicted_values, label="Predicted Values", color="red", s=10)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("ViT: Predicted and True Values")

    plt.legend()
    plt.savefig("../results/plots/ViT_results.png")
    plt.close()

    ####################################### Plot training loss #############################################

    # Load the training loss
    loss = []
    print(len(trainer.state.log_history))
    for i in range(0, len(trainer.state.log_history)):
        try:
            training_loss_per_epoch = trainer.state.log_history[i]['loss']
            loss.append(training_loss_per_epoch)
        except KeyError:
            # 'loss' key doesn't exist, skip this iteration
            continue

    # Convert the loss list to a numpy array
    loss_array = np.array(loss)

    # Calculate the step size to select uniformly distant points
    step_size = len(loss_array) // 90

    # Select uniformly distant points from the loss array
    selected_points = loss_array[::step_size]

    # Plot the training loss
    plt.figure()
    plt.plot(selected_points)
    #plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("ViT: Training Loss vs Epoch")
    plt.savefig("../results/plots/ViT_train_loss.png")
    plt.close()

    ############################# Compute the accuracy within the defined threshold #############################

    # Load the predictions from the CSV file
    df = pd.read_csv("../results/ViT_Tuned_Predictions.csv")

    # Extract the predicted and true values
    predicted_values = df["Predicted Values"]
    true_values = df["True Values"]

    threshold = 0.15
    correct = 0
    for i in range(0, len(true_values)):
        if np.abs(predicted_values[i] - true_values[i]) < threshold * np.abs(true_values[i]):
            correct += 1

    accuracy = correct / len(true_values)
    print(accuracy)