import time
import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from Model.Model import SimpleCNN
from Utilities.Images import ImageDataset

torch.manual_seed(42)

IMG_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 15
NUM_CLASSES = 4
LEARNING_RATE = 0.001

CLASS_NAMES = ['Egyptian cat', 'banana', 'African elephant', 'mountain bike']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class statistics
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc, class_correct, class_total


print("=" * 80)
print("ECE 49595 Computer Vision - Assignment 2: Image Classification")
print("=" * 80)
print()

# Get the directory where the script is located
script_dir = Path(__file__).parent
data_dir = script_dir / 'Datasets'
train_file = script_dir / 'Datasets' / 'train.txt'
test_file = script_dir / 'Datasets' / 'test.txt'

print(f"Data directory: {data_dir}")
print(f"Training file: {train_file}")
print(f"Test file: {test_file}")
print()

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create datasets
print("Loading datasets...")
train_dataset = ImageDataset(train_file, data_dir, transform=train_transform,
                             img_size=IMG_SIZE)
test_dataset = ImageDataset(test_file, data_dir, transform=test_transform,
                            img_size=IMG_SIZE)

# Report any missing files
train_dataset.report_missing_files()
test_dataset.report_missing_files()

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print()

# Initialize model, loss function, and optimizer
print("Initializing model...")
model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print()

# Training
print("=" * 80)
print("Starting Training")
print("=" * 80)

start_time = time.time()
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    train_loss, train_acc = train_model(model, train_loader, criterion,
                                        optimizer, device)
    # Evaluate
    test_loss, test_acc, _, _ = evaluate_model(model, test_loader,
                                               criterion, device)

    # Learning rate scheduling
    scheduler.step()
    epoch_time = time.time() - epoch_start

    # Print progress
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    print()

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc

total_time = time.time() - start_time

print("=" * 80)
print("Training Complete!")
print("=" * 80)
print(f"Total training time: {total_time:.2f} seconds "
      f"({total_time/60:.2f} minutes)")
print(f"Best test accuracy: {best_acc:.2f}%")
print()

# Final evaluation with per-class accuracy
print("=" * 80)
print("Final Test Results")
print("=" * 80)

test_loss, test_acc, class_correct, class_total = evaluate_model(
    model, test_loader, criterion, device
)

print(f"Overall Test Accuracy: {test_acc:.2f}%")
print(f"Overall Test Loss: {test_loss:.4f}")
print()

print("Per-Class Accuracy:")
print("-" * 50)
for i in range(NUM_CLASSES):
    if class_total[i] > 0:
        acc = 100. * class_correct[i] / class_total[i]
        print(f"  {CLASS_NAMES[i]:20s}: {acc:.2f}% "
              f"({class_correct[i]}/{class_total[i]})")
    else:
        print(f"  {CLASS_NAMES[i]:20s}: No samples")
print()

# Success/Failure message
print("=" * 80)
if test_acc > 50.0:
    print(f"✓ SUCCESS: Achieved {test_acc:.2f}% accuracy "
          f"(target: >50%)")
    print(f"✓ Training completed in {total_time:.2f}s "
          f"(target: <300s)")
else:
    print(f"✗ FAILED: Achieved {test_acc:.2f}% accuracy (target: >50%)")
print("=" * 80)
