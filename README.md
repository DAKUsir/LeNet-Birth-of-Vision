# LeNet — Glaucoma Detection (mini-project)

**Dataset (Kaggle):** [https://www.kaggle.com/datasets/chauhanadityacse/glaucoma-detection](https://www.kaggle.com/datasets/chauhanadityacse/glaucoma-detection)

---

## Project summary

This repository/notebook trains a compact **LeNet-style CNN** to detect **glaucoma** from retinal (fundus) images. The pipeline included here is lightweight and suitable for rapid experimentation on Kaggle or a local GPU machine.

**Key features**

* Data loading from folder-structured dataset
* Train / validation split
* Data augmentation (rotation, horizontal flip)
* Weighted cross-entropy loss for class imbalance
* LeNet variant adapted to 64×64 grayscale input
* Training & validation graphs (loss & accuracy)
* Save / load model
* Single-image prediction helper

---

## Quick links

* Kaggle dataset used: [https://www.kaggle.com/datasets/chauhanadityacse/glaucoma-detection](https://www.kaggle.com/datasets/chauhanadityacse/glaucoma-detection)

---

## Requirements

* Python 3.8+
* PyTorch (1.7+)
* torchvision
* Pillow (PIL)
* numpy, matplotlib

Kaggle notebooks already include these packages.

---

## Folder structure (recommended)

```
/notebook.ipynb            # main Kaggle notebook
/README.md                 # this file
/models/
  lenet_glaucoma_adv.pth   # saved model weights
```

---

## LeNet (brief teaching note)

LeNet-5 is a classic convolutional neural network introduced by Yann LeCun (1998). It's a small, fast architecture good for learning and experimentation on low-resolution images.

**Classic flow (conceptual)**

```
Input (1x32x32) -> Conv(6,5x5) -> Pool(2x2) -> Conv(16,5x5) -> Pool -> FC(120) -> FC(84) -> FC(classes)
```

**This project changes**:

* Input size increased to **64×64** to retain more retinal detail.
* Grayscale input (1 channel).
* Weighted loss to handle class imbalance.

---

## How to run (Kaggle notebook / local)

Below are condensed steps you can paste into a notebook cell. Each item is also included with more detail in the notebook.

### 1) Imports & paths

```python
import os, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Point this to the dataset path in Kaggle
GLAUCOMA_PATHS = [
  '/kaggle/input/glaucoma-detection/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train/Glaucoma_Negative',
  '/kaggle/input/glaucoma-detection/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train/Glaucoma_Positive',
  '/kaggle/input/glaucoma-detection/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Validation/Glaucoma_Negative',
  '/kaggle/input/glaucoma-detection/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Validation/Glaucoma_Positive'
]
```

### 2) Collect samples

```python
all_samples = []
for path in GLAUCOMA_PATHS:
    label = 0 if 'Negative' in path else 1
    for f in os.listdir(path):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            all_samples.append((os.path.join(path,f), label))
print('Total samples:', len(all_samples))
```

### 3) Transforms and Dataset

```python
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class GlaucomaDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
```

### 4) Split and dataloaders

```python
dataset = GlaucomaDataset(all_samples, transform=transform_train)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# set validation transform
val_dataset.dataset.transform = transform_val
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
```

### 5) Model (LeNet variant)

```python
class LeNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*13*13, 120) # for 64x64 input
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet().to(device)
```

### 6) Weighted loss & optimizer

```python
labels = [lbl for _, lbl in all_samples]
counts = np.bincount(labels)
class_weights = 1.0 / (counts + 1e-6)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### 7) Training loop (with metrics)

```python
epochs = 20
train_losses, val_losses, val_accs = [], [], []
for epoch in range(epochs):
    model.train()
    running = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        running += loss.item()
    train_losses.append(running/len(train_loader))
    # validation
    model.eval(); vloss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            vloss += loss.item()
            _, pred = out.max(1)
            total += labels.size(0); correct += (pred==labels).sum().item()
    val_losses.append(vloss/len(val_loader))
    val_accs.append(100*correct/total)
    print(f"Epoch {epoch+1}/{epochs} | TrainLoss {train_losses[-1]:.4f} | ValLoss {val_losses[-1]:.4f} | ValAcc {val_accs[-1]:.2f}%")
```

### 8) Graphs

```python
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.plot(train_losses,label='train'); plt.plot(val_losses,label='val'); plt.title('Loss'); plt.legend()
plt.subplot(1,2,2); plt.plot(val_accs,label='val_acc'); plt.title('Val Accuracy'); plt.legend()
plt.show()
```

### 9) Save & load

```python
torch.save(model.state_dict(), 'lenet_glaucoma_adv.pth')
# load
model = LeNet().to(device)
model.load_state_dict(torch.load('lenet_glaucoma_adv.pth', map_location=device))
model.eval()
```

### 10) Single image prediction helper

```python
from torchvision import transforms
predict_transform = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

def predict_single(image_path, model, device):
    img = Image.open(image_path).convert('RGB')
    inp = predict_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(out.argmax(1).item())
    return pred, prob
```

---

## Troubleshooting & tips

* **Predictions collapsed to single class**: train longer, use data augmentation, weighted loss or oversampling.
* **OOM on GPU**: reduce batch size (e.g., 16 → 8 → 4) and/or use `num_workers=0`.
* **Better accuracy**: use transfer learning (ResNet18 / EfficientNet) on 224×224 images.
* **Explainability**: add Grad-CAM to visualise model attention.


---

## Contact

Linkedin : https://www.linkedin.com/in/aditya-chauhan-async/
---

*End of README.*
