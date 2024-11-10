import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from model import MNISTNet
from tqdm import tqdm
import os

# Clear previous logs
def clear_logs():
    """Clear previous training and test logs"""
    if os.path.exists('training_data.json'):
        os.remove('training_data.json')
    if os.path.exists('test_results.json'):
        os.remove('test_results.json')
    print("Previous logs cleared")

clear_logs()

# Set random seed for reproducibility
torch.manual_seed(42)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Data preparation
print("Loading datasets...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512)
print(f"Dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# Initialize model, loss function, and optimizer
print("Initializing model...")
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Evaluating', leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(data_loader), correct / total

def save_training_data():
    with open('training_data.json', 'w') as f:
        json.dump(history, f)

# Training loop
print("\nStarting training...")
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Update progress bar description
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    train_loss = train_loss / len(train_loader)
    train_acc = correct / total
    
    # Validation
    print("\nRunning validation...")
    val_loss, val_acc = evaluate(model, test_loader)
    
    # Update history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # Save training data for visualization
    save_training_data()
    
    print(f'\nEpoch {epoch+1} Summary:')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n')

print("Training completed!")

# Generate test results on 10 random images
print("Generating test results...")
def generate_test_results():
    model.eval()
    test_results = {
        'images': [],
        'predictions': [],
        'labels': []
    }
    
    # Get 10 random test samples
    indices = np.random.choice(len(test_dataset), 10, replace=False)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='Processing test images'):
            image, label = test_dataset[idx]
            
            # Get prediction
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            
            # Convert tensor to image
            img = image.squeeze().numpy()
            buf = io.BytesIO()
            plt.figure(figsize=(2, 2))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            test_results['images'].append(img_str)
            test_results['predictions'].append(int(pred))
            test_results['labels'].append(int(label))
    
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f)

# Generate and save test results
generate_test_results()
print("Test results saved. All done!")