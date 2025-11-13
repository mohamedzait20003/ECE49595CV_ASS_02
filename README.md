# ECE 49595CV Assignment 2: Image Classification

## Overview

This assignment implements a PyTorch-based Convolutional Neural Network (CNN) for classifying images into 4 classes:

- Egyptian cat (class 0)
- banana (class 1)  
- African elephant (class 2)
- mountain bike (class 3)

## Features

- **AlexNet-ResNet Hybrid Architecture**: 5 convolutional layers with ResNet improvements
- **Skip Connections**: Better gradient flow for deeper training
- **Batch Normalization**: Stable and faster training convergence
- **Data augmentation**: Random flips, rotations, and color jittering
- **Dropout regularization**: Progressive dropout to prevent overfitting
- **Balanced Dataset**: Proper 4-class distribution after dataset correction

## Model Architecture

The improved SimpleCNN model features:

- **5 Convolutional Layers** (AlexNet-style) with ResNet improvements:
  - Conv1: 3→64 channels (5x5 kernel)
  - Conv2: 64→192 channels (3x3 kernel)  
  - Conv3: 192→384 channels (3x3 kernel)
  - Conv4: 384→256 channels (3x3 kernel) + **Skip Connection**
  - Conv5: 256→256 channels (3x3 kernel) + **Skip Connection**
- **Batch Normalization** after each convolution
- **3 Fully Connected Layers**: 2304→1024→256→4 with BatchNorm
- **Progressive Dropout**: 0.5→0.3→0.2 for better generalization

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- Pillow
- NumPy
- tqdm (for progress bars)
- matplotlib (for visualization)



Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

```
Datasets/
├── train.txt           # Training image paths with class labels
├── test.txt            # Test image paths with class labels
├── n02124075/          # Egyptian cat images  
├── n07753592/          # banana images
├── n02504458/          # African elephant images
└── n03792782/          # mountain bike images
```

## Dataset

The dataset consists of:

- 4,152 training images
- 1,039 test images  
- Balanced distribution across 4 classes

Dataset files:
- `Datasets/train.txt`: Training image paths with class labels
- `Datasets/test.txt`: Test image paths with class labels

## Expected Performance

- **Test Accuracy**: 60-75% (realistic expectation for balanced 4-class dataset)
- **Training**: 15 epochs with Adam optimizer and StepLR scheduler  
- **Memory Usage**: ~200MB for model + batch processing

## Usage

### Training

```bash
python main.py
```

Or use the interactive notebook:

```bash
jupyter notebook Notebooks/notebook.ipynb
```

### Windows PowerShell

```powershell
.\run.ps1
```

### Linux/Mac

```bash
chmod +x run
./run
```

## Output

The training script will:

1. Load and preprocess the training and test datasets
2. Train the CNN for 15 epochs with progress bars
3. Display training and test accuracy for each epoch
4. Show final test accuracy and per-class performance
5. Save the trained model

## Project Files

- `main.py` - Main PyTorch training script
- `Model/Model.py` - ResNet-improved CNN architecture
- `Utilities/Images.py` - Custom dataset class with image loading
- `Notebooks/notebook.ipynb` - Interactive training notebook
- `requirements.txt` - Python dependencies
- `run.ps1` - PowerShell execution script
- `run` - Bash execution script
- `Makefile` - Build automation

## Implementation Details

- **Image size**: 64x64 pixels for efficient training
- **Batch size**: 64 images per batch
- **Optimizer**: Adam with learning rate 0.001
- **Scheduler**: StepLR (decay every 5 epochs)  
- **Loss function**: Cross-entropy loss
- **Data augmentation**: Horizontal flip, rotation, color jitter
- **Normalization**: ImageNet mean and std

## Troubleshooting

**Low accuracy (< 30%)**
- Check dataset balance with `fix_dataset.py`
- Verify image paths in train.txt/test.txt
- Ensure proper data preprocessing

**Memory issues**
- Reduce batch size in main.py
- Enable CUDA if available for GPU memory

**Missing dependencies**
- Run `pip install -r requirements.txt`
- Install PyTorch with CUDA support if needed

## Author

Mohamed Mahmoud Zaitoun for Purdue University - ECE 49595CV Fall 2025
