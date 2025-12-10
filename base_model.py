#!/cs4701-venv/bin/python
# coding: utf-8

# Import Libraries
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import kagglehub
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim 
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load Data
def download_dataset():
    # Download latest version of data
    path = kagglehub.dataset_download("lexset/synthetic-asl-alphabet")
    print("Path to dataset files:", path)
    return path

def preprocess_dataset(path):
    # Initialize transformation: format as tensor, normalize
    transform_tr = transforms.Compose([
        #transforms.RandomRotation(15), # randomly rotates some images by up to 15 degrees
        #transforms.ColorJitter(brightness=0.2, contrast=0.2), # randomly changes the brightness, contrast, saturation, and hue of an image
        #transforms.RandomGrayscale(p=0.1), # randomly grayscales some images
        #transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)), # new transform
        #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # new transform
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5] 
            )
        ])

    # Load data folder
    dat = torchvision.datasets.ImageFolder(
        root = path + "/Train_Alphabet",
        transform = transform_tr)
    
    # Subset Data
    # Drop J and Z
    to_drop = ['J','Z']

    dat.imgs = dat.imgs[0:10*900] + [i for i in dat.imgs[11*900:26*900]]
    dat.samples = dat.samples[0:10*900] + [(i,l-1) for i,l in dat.samples[11*900:26*900]]
    dat.targets = dat.targets[0:10*900] + [i-1 for i in dat.targets[11*900:26*900]]
    dat.classes = [i for i in dat.classes if i not in to_drop ]

    dat.class_to_idx = {
        'A': 0,
        'B': 1,
        'Blank': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7,
        'H': 8,
        'I': 9,
        'K': 10,
        'L': 11,
        'M': 12,
        'N': 13,
        'O': 14,
        'P': 15,
        'Q': 16,
        'R': 17,
        'S': 18,
        'T': 19,
        'U': 20,
        'V': 21,
        'W': 22,
        'X': 23,
        'Y': 24
    }
    return dat

# test-train-val split
def test_train_split(dat, BATCH_SIZE=64):

    len(dat.targets) == len(dat.samples) and len(dat.targets) == len(dat.imgs) and len(dat.imgs) == len(dat) and len(dat) == 900*25
    len(dat.classes) == 25
    # Split data
    ratio_tr = 0.8
    #ratio_val = 0.2

    size_tr = int(len(dat) * ratio_tr)
    size_val = len(dat) - size_tr

    dat_tr, dat_val = random_split(dat, [size_tr, size_val])

    # Load image data from folder with some going into training and some to validation
    loader_tr = DataLoader(dat_tr, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(dat_val, batch_size=BATCH_SIZE, shuffle = False)

    return loader_tr, loader_val

#  Initialize Conv NN
def initialize_model(num_classes=25):
    # Note: output channels are arbitrary
    conv_model = nn.Sequential(
            # 3 input channels for RGB
            nn.Conv2d(3, 10, kernel_size = 3, stride = 2), 
            nn.ReLU(),
            nn.MaxPool2d(2), 
            # output dim ~128
            
            nn.Conv2d(10, 20, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # output dim ~32

            nn.Conv2d(20, 25, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # output dim 7

            nn.Flatten(),
            nn.Linear(25*7*7, 400),
            nn.ReLU(),

            nn.Linear(400, num_classes)
        )
    return conv_model

def train_model(conv_model, loader_tr, num_epochs=2):

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(conv_model.parameters())

    # Fit model on data
    for epoch in range(num_epochs):
        count = 0
        curr_loss = 0
        for i, data in tqdm(enumerate(loader_tr)):
            images, labels = data

            optimizer.zero_grad()

            outputs = conv_model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            count += 1
            curr_loss += loss.item()
            if count % 25 == 0:
                print(f"Epoch {epoch + 1}, Step {count}, Loss: {curr_loss}")
                curr_loss = 0


# Evaluate Model
def evaluate_model(conv_model, loader_tr, loader_val, dat, mode="Validation"):
    correct = 0
    total = 0

    for i, data in tqdm(enumerate(loader_tr)):
        images, labels = data

        outputs = conv_model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.shape[0]

    print(f"{mode} Accuracy: " + str(correct / total))

# Per class accuracy
def per_class_accuracy(dat, loader_val, conv_model):
    reverse_map = {dat.class_to_idx[c]: c for c in dat.class_to_idx.keys()}
    correct = {c:0 for c in dat.classes}
    total = {c:0 for c in dat.classes}

    conv_model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader_val)):
            images, labels = data

            outputs = conv_model(images)
            _, predicted = torch.max(outputs, 1)

            for lab, pred in zip(labels, predicted):
                if lab == pred:
                    correct[reverse_map[lab.item()]] += 1
                total[reverse_map[lab.item()]] += 1
    class_accuracy = {c: correct[c] / total[c] if total[c] > 0 else 0 for c in total.keys()}

    for c,acc in class_accuracy.items():
        print(f"{c} Accuracy: {acc:.2%}")
    
    return class_accuracy

# def plot_confusion_matrix(true_labels, pred_labels, class_names):
#     cm = confusion_matrix(true_labels, pred_labels)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#     disp.plot(cmap=plt.cm.Blues)
#     plt.show()


def main():
    path = download_dataset()
    dat = preprocess_dataset(path)
    loader_tr, loader_val = test_train_split(dat)
    conv_model = initialize_model(num_classes=len(dat.classes))
    train_model(conv_model, loader_tr, num_epochs=2)
    evaluate_model(conv_model, loader_tr, loader_val, dat, mode="Training")
    evaluate_model(conv_model, loader_tr, loader_val, dat, mode="Validation")

    print("\nCaclulating per-class accuracy:\n")
    per_class_accuracy(dat, loader_val, conv_model)

if __name__ == "__main__":
    main()