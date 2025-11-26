import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import segmentation_models_pytorch as smp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

img_dir = "train/"         
mask_dir = "train_masks/"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
num_epochs = 22
lr = 1e-4
img_size = (224, 224)
test_size = 0.1  

class XraySegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg','.png'))])
        assert len(self.img_files) == len(self.mask_files), "Number of images and masks must match"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(img_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0

        mask_path = os.path.join(mask_dir, self.mask_files[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        return img_tensor, mask_tensor

dataset = XraySegDataset(img_dir, mask_dir)
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)

dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss = nn.BCEWithLogitsLoss()

def combined_loss(preds, targets):
    return dice_loss(preds, targets) + bce_loss(preds, targets)

criterion = combined_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training")
    for imgs, masks in loop:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "unet_resnet50_xray_seg.pth")
        print("Saved best model.")

print("Training complete!")
