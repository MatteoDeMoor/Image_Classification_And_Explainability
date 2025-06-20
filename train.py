import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from multiprocessing import freeze_support

# Config 
DATA_DIR            = Path("data")
TRAIN_DIR           = DATA_DIR / "train"
VAL_DIR             = DATA_DIR / "val"
MODEL_DIR           = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
BATCH_SIZE          = 32
NUM_EPOCHS_HEAD     = 5
NUM_EPOCHS_FINETUNE = 5
LEARNING_RATE_HEAD  = 1e-3
LEARNING_RATE_ALL   = 1e-4
IMAGE_SIZE          = 224
NUM_WORKERS         = 4  # you can set to 0 if you still run into issues

# Device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Helper functions 
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc="Train", leave=False)
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct/total)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(loader, desc="Validate", leave=False)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=correct/total)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    # Necessary for Windows multiprocessing
    freeze_support()

    # Data Transforms & Loaders 
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    num_classes = len(train_ds.classes)
    print(f"Found {num_classes} classes: {train_ds.classes}")

    # Model: use torchvision’s ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.to(DEVICE)

    # Train classifier head only 
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze final fully-connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE_HEAD)

    best_val_acc = 0.0
    print(" Training head only ")
    for epoch in range(1, NUM_EPOCHS_HEAD + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS_HEAD}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = validate(model, val_loader, criterion)
        print(f"  Train   loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Validate loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_head.pth")

    # Unfreeze & fine-tune all layers 
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_ALL)
    print(" Fine-tuning entire model ")
    for epoch in range(1, NUM_EPOCHS_FINETUNE + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS_FINETUNE}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = validate(model, val_loader, criterion)
        print(f"  Train   loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Validate loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to {MODEL_DIR / 'best_model.pth'}")
