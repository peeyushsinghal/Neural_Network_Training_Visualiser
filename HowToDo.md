# CNN MNIST Training Visualization

This project implements a CNN for MNIST classification with real-time training visualization.

## Setup and Requirements

1. Install required packages:
```bash
pip install torch torchvision flask numpy matplotlib
```

2. Project Structure:
```
├── main.py           # Main training script
├── model.py          # CNN model definition
├── server.py         # Flask server for visualization
└── templates
    └── index.html    # Training visualization page
```

## Model Architecture

The CNN consists of 4 convolutional layers:
- Layer 1: 1 → 32 channels, 3x3 kernel, ReLU, MaxPool
- Layer 2: 32 → 64 channels, 3x3 kernel, ReLU, MaxPool
- Layer 3: 64 → 128 channels, 3x3 kernel, ReLU, MaxPool
- Layer 4: 128 → 256 channels, 3x3 kernel, ReLU, AdaptiveAvgPool
- Final: Flatten + Linear layer (256 → 10)

## Running the Project

1. Start the Flask server first:
```bash
python server.py
```

2. In a new terminal, start the training:
```bash
python main.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## What to Expect

1. Real-time Training Visualization:
   - Loss curves showing training and validation loss
   - Accuracy curves showing training and validation accuracy
   - Updates every 5 seconds automatically

2. Training Parameters:
   - Number of epochs: 10
   - Batch size: 64
   - Optimizer: Adam (lr=0.001)
   - Loss function: CrossEntropyLoss

3. After Training:
   - 10 random test images will be displayed
   - For each image, you'll see:
     - The original image
     - Predicted digit
     - Actual digit

## Troubleshooting

1. If you see "No module found" errors:
   - Ensure all required packages are installed
   - Check your Python environment

2. If visualization doesn't update:
   - Verify both server.py and main.py are running
   - Check browser console for errors
   - Ensure training_data.json is being created/updated

3. If CUDA errors occur:
   - The code will automatically fall back to CPU if CUDA is not available
   - No changes needed in the code

## Files Description

- `main.py`: Contains the training loop, data loading, and test result generation
- `model.py`: Defines the CNN architecture
- `server.py`: Flask server for serving the visualization webpage
- `index.html`: Frontend visualization interface
- `training_data.json`: Real-time training metrics (auto-generated)
- `test_results.json`: Test predictions and images (generated after training)