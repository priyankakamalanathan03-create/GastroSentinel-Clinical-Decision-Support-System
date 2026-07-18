import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from model import get_model 

def train_gastro_sentinel():
    # 1. Configuration
    EPOCHS = 20 
    BATCH_SIZE = 32
    LR = 1e-4 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Engine Started ---")
    print(f"Target Hardware: {DEVICE}")

    # 2. Clinical Data Augmentation
    # We use these to make the AI robust against different camera angles
    transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 3. Data Loaders (70/15 split)
    train_set = datasets.ImageFolder('data/processed/train', transform['train'])
    val_set = datasets.ImageFolder('data/processed/val', transform['val'])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Model, Optimizer, and Clinical Loss Function
    model = get_model(num_classes=8).to(DEVICE)
    
    # Label Smoothing (0.1) prevents the model from over-fitting to noisy medical labels
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW provides better regularization than standard Adam
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    # Scheduler: Automatically lowers the learning rate if the model stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # 5. Training Loop
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
        epoch_val_acc = val_correct / val_total
        epoch_val_loss = val_loss / len(val_loader)
        
        print(f"Summary: Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # Deep Logic: Step the scheduler based on validation loss
        scheduler.step(epoch_val_loss)

        # Save ONLY the best clinical version
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'models/gastro_sentinel_best.pth')
            print(f"*** New Best Model Saved (Accuracy: {best_val_acc:.4f}) ***")

if __name__ == "__main__":
    train_gastro_sentinel()