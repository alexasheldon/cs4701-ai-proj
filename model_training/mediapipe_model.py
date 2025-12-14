import torch
from tqdm import tqdm
import kagglehub
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim 
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils, hands

# Load Data
def download_dataset_synthetic():
    # Download latest version of data
    path = kagglehub.dataset_download("lexset/synthetic-asl-alphabet")
    print("Path to dataset files:", path)
    return path

def download_dataset():
    # Download latest version
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    print("Path to dataset files:", path)
    return path

def initialize_transformation(): 
    # Initialize transformation: format as tensor
    return transforms.Compose([
        transforms.ToTensor()
        ])

def initialize_transformation_noise(): 
    # Initialize transformation: format as tensor
    return transforms.Compose([
        transforms.RandomRotation(15), # randomly rotates some images by up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # randomly changes the brightness, contrast, saturation, and hue of an image
        transforms.RandomGrayscale(p=0.1), # randomly grayscales some images
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)), # new transform
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # new transform
        transforms.ToTensor(),
        ])

def assign_mapping(dat):
    dat.class_to_idx = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'K': 9,
        'L': 10,
        'M': 11,
        'N': 12,
        'O': 13,
        'P': 14,
        'Q': 15,
        'R': 16,
        'S': 17,
        'T': 18,
        'U': 19,
        'V': 20,
        'W': 21,
        'X': 22,
        'Y': 23
    }

def preprocess_dataset_synthetic(path, transform_tr):
    # Load data folder
    dat = torchvision.datasets.ImageFolder(
        root = path + "/Train_Alphabet",
        transform = transform_tr)
    
    # Drop Blank, J, and Z
    to_drop = ['Blank','J','Z']

    dat.imgs = dat.imgs[0:2*900] + [i for i in dat.imgs[3*900:10*900]] + [i for i in dat.imgs[11*900:26*900]]
    dat.samples = dat.samples[0:2*900] + [(i,l-1) for i,l in dat.samples[3*900:10*900]] + [(i,l-2) for i,l in dat.samples[11*900:26*900]]
    dat.targets = dat.targets[0:2*900] + [i-1 for i in dat.targets[3*900:10*900]] + [i-2 for i in dat.targets[11*900:26*900]]
    dat.classes = [i for i in dat.classes if i not in to_drop ]

    assign_mapping(dat)

    return dat

def preprocess_dataset(path, transform_tr):
    # Load data folder
    dat = torchvision.datasets.ImageFolder(
        root = path + "/asl_alphabet_train/asl_alphabet_train",
        transform = transform_tr)
    
    # Drop Blank, J, and Z
    to_drop = ['J','Z','nothing','del','space']

    dat.imgs = dat.imgs[0:9*3000] + [i for i in dat.imgs[10*3000:25*3000]]
    dat.samples = dat.samples[0:9*3000] + [(i,l-1) for i,l in dat.samples[10*3000:25*3000]]
    dat.targets = dat.targets[0:9*3000] + [i-1 for i in dat.targets[10*3000:25*3000]]
    dat.classes = [i for i in dat.classes if i not in to_drop ]

    assign_mapping(dat)

    return dat


def test_train_split(dat, ratio_tr = 0.8, BATCH_SIZE=64):
    # Split data
    size_tr = int(len(dat) * ratio_tr)
    size_val = len(dat) - size_tr

    dat_tr, dat_val = random_split(dat, [size_tr, size_val])

    # Load image data from folder with some going into training and some to validation
    loader_tr = DataLoader(dat_tr, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(dat_val, batch_size=BATCH_SIZE, shuffle = False)

    return loader_tr, loader_val

def display_image(loader_tr, i=0):
    # Present image
    images, labels = next(iter(loader_tr))

    img = images[i].squeeze()
    img = img.permute(1, 2, 0)  
    label = labels[i]

    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")


# Format Data
def normalize_landmarks(res):
    # Convert landmarks to numpy array
    lm = np.array([[lmk.x, lmk.y, lmk.z] for lmk in res[0]])

    # Translate landmarks so that wrist is at origin
    lm -= lm[0]
    # Calculate scale (max distance from origin)
    scale = np.linalg.norm(lm[9]) if np.linalg.norm(lm[9]) > 1e-6 else np.max(np.linalg.norm(lm, axis=1))
    lm /= (scale + 1e-8)
    return lm

def process_batch(images, labels, detector):
    imgs_np = (images.permute(0,2,3,1) * 255).byte().cpu().numpy()

    # Iterate through images in batch
    dat_batch = []
    labels_batch = []
    for j, img_np in enumerate(imgs_np):
        # Convert to mediapipe format
        img_np = np.ascontiguousarray(img_np)
        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

        # Get embeddings
        res = (detector.detect(img_mp)).hand_landmarks           

        # Add embeddings to result tensor if detected
        if res:
            dat_norm = torch.tensor(normalize_landmarks(res), dtype=torch.float32)
            dat_norm = dat_norm.view(1,dat_norm.size(0),dat_norm.size(1))
                
            dat_batch.append(dat_norm)
            labels_batch.append(labels[j])

    return dat_batch, labels_batch

def get_embeddings(loader):
    base_options = python.BaseOptions(model_asset_path='../hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options)
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize result
    dat_mp = []
    labels_mp = []

    # Call thread on each batch
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_batch, images, labels, detector) for images, labels in tqdm(loader)}

        # Combine output
        for f in tqdm(as_completed(futures), total=len(futures)):
            dat_batch, labels_batch = f.result()
            dat_mp.append(torch.cat(dat_batch, dim=0))
            labels_mp.append(torch.tensor(labels_batch))

    return dat_mp, labels_mp

def approximate_undetected(data_loader, dat_mp):
    total = 0
    for i in dat_mp:
        total += i.shape[0]

    total_loader = len(data_loader)*(data_loader.batch_size)
    undetected = total_loader - total

    print("Approximate number of undetected images: " + str(undetected))
    print("Approximate percentage of images undetected: " + str(round((undetected/total_loader) * 100, 2)) + "%")


# Fit Model
def initialize_model(num_classes = 24):
    # Initialize MLP
    return nn.Sequential(
            nn.Linear(21*3, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, num_classes),
        )

def train_model(mlp_model, dat_mp, labels_mp, NUM_EPOCH=50):
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters())

    # Fit model on data
    count = 0
    for epoch in tqdm(range(NUM_EPOCH)):
        curr_loss = 0
        for i, hand_lm in enumerate(dat_mp): # enumerate in batch size
            labels_batch = labels_mp[i]
            hand_lm = hand_lm.view(hand_lm.size(0), -1)     # flatten embedding

            optimizer.zero_grad()

            outputs = mlp_model(hand_lm)
            loss = criterion(outputs, labels_batch)
            
            loss.backward()
            optimizer.step()
            
            curr_loss += loss.item()

        count += 1
        if count % 5 == 0:
            print("loss: " + str(curr_loss))
            curr_loss = 0


# Evaluation
def evaluate_model(mlp_model, dat_mp, labels_mp, mode = "Validation"):
    correct = 0
    total = 0

    for i, hand_lm in tqdm(enumerate(dat_mp)):
        labels_batch = labels_mp[i]
        hand_lm = hand_lm.view(hand_lm.size(0), -1)

        outputs = mlp_model(hand_lm)
        _, predicted = torch.max(outputs, 1)
        
        correct += (predicted == labels_batch).sum().item()
        total += labels_batch.shape[0]

    accuracy = correct / total
    print("{mode} Accuracy: " + str(accuracy))

    return accuracy

def per_class_accuracy(dat, mlp_model, dat_mp, labels_mp):
    reverse_map = {dat.class_to_idx[c]: c for c in dat.class_to_idx.keys()}
    correct = {c:0 for c in dat.class_to_idx.keys()}
    total = {c:0 for c in dat.class_to_idx.keys()}

    mlp_model.eval()
    with torch.no_grad():
        for i, hand_lm in tqdm(enumerate(dat_mp)):
            labels_batch = labels_mp[i]
            hand_lm = hand_lm.view(hand_lm.size(0), -1)

            outputs = mlp_model(hand_lm)
            _, predicted = torch.max(outputs, 1)

            for lab, pred in zip(labels_batch, predicted):
                if lab == pred:
                    correct[reverse_map[lab.item()]] += 1
                total[reverse_map[lab.item()]] += 1
    class_accuracy = {c: correct[c] / total[c] if total[c] > 0 else 0 for c in total.keys()}

    for c,acc in class_accuracy.items():
        print(f"{c} Accuracy: {acc:.2%}")

    return class_accuracy

def get_pred_labels(mlp_model, dat_mp):
    pred_labels = []

    for hand_lm in tqdm(dat_mp):
        hand_lm = hand_lm.view(hand_lm.size(0), -1)

        outputs = mlp_model(hand_lm)
        _, predicted = torch.max(outputs, 1)
        pred_labels.append(predicted)

    return pred_labels

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)

    for text in disp.text_.ravel():
        text.set_fontsize(7)

    plt.show()

def get_wrong_predictions(letter, pred_labels, true_labels, dat, reverse_map):
    label_num = dat.class_to_idx[letter]
    letter_preds = pred_labels[np.where(true_labels == label_num)[0]]
    wrong_preds = letter_preds[torch.nonzero(letter_preds != label_num).squeeze()]

    vals, counts = torch.unique(wrong_preds, return_counts=True)

    mapped = []
    for x in vals:
        mapped.append(reverse_map[int(x)])

    res = pd.DataFrame(np.array(counts), mapped, columns=[letter])
    return res.sort_values(by=letter, ascending = False)

# Save/Load Model
def save_model(mlp_model, model_name):
    torch.save(mlp_model, model_name+".pt")

def load_model(model_name):
    return torch.load(model_name+".pt")