'''This script splits the PepNet dataset into train/val/test sets *before* applying
data augmentation, then augments only the training images.

Augmenting first and splitting afterwards (the original approach) leaks information:
each peptide's original image and its augmented copy carry the same label and are highly
correlated, so if they land on opposite sides of a random split, the model is effectively
evaluated on near-duplicates of what it trained on, inflating validation/test performance.
Splitting by peptide first guarantees a peptide's images (original + augmented) always stay
together on one side of the split.'''

import h5py
import numpy as np
from torchvision import transforms

SEED = 69
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# The remainder (1 - TRAIN_FRAC - VAL_FRAC) goes to the test split.

# Define the data augmentation transformations
data_transforms = transforms.Compose([
    transforms.ToPILImage(),                  # Convert numpy array to PIL Image
    transforms.RandomHorizontalFlip(),       # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),         # Randomly flip the image vertically
    transforms.RandomRotation(degrees=15),   # Randomly rotate the image by a maximum of 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
])

# Path to the HDF5 file
hdf5_file_path = '../data/PepNet_data.h5'

# Open the HDF5 file in read mode
with h5py.File(hdf5_file_path, 'r') as f:
    images = np.array(f['images'][:]).astype(np.uint8)
    labels = np.array(f['permeability'][:])

# Split peptides into train/val/test *before* augmenting.
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(images))

n_train = int(len(indices) * TRAIN_FRAC)
n_val = int(len(indices) * VAL_FRAC)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]


def save_split(path, split_images, split_labels):
    with h5py.File(path, 'w') as f:
        f.create_dataset('images', data=np.asarray(split_images, dtype='uint8'))
        f.create_dataset('permeability', data=np.asarray(split_labels, dtype=float))


# Validation and test sets stay unaugmented, so evaluation reflects real, unseen peptides.
save_split('../data/PepNet_val.h5', images[val_idx], labels[val_idx])
save_split('../data/PepNet_test.h5', images[test_idx], labels[test_idx])

# Augment only the training split: keep the original image plus one randomly
# augmented copy for each training peptide.
augmented_images = []
augmented_labels = []

for image, label in zip(images[train_idx], labels[train_idx]):
    # Append original image and label
    augmented_images.append(image)
    augmented_labels.append(label)

    # Apply data augmentation transformations
    augmented_image = data_transforms(image)

    # Convert augmented image to numpy array and append to the list
    augmented_images.append(np.array(augmented_image))

    # Append the label again for the augmented image
    augmented_labels.append(label)

save_split('../data/PepNet_train_augmented.h5', augmented_images, augmented_labels)

print(
    f"Split {len(images)} peptides into "
    f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test."
)
print("Train split (original + augmented) saved to ../data/PepNet_train_augmented.h5")
print("Val split (unaugmented) saved to ../data/PepNet_val.h5")
print("Test split (unaugmented) saved to ../data/PepNet_test.h5")
