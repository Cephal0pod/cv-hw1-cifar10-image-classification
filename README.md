# CIFAR-10 Image Classification

This repository contains an implementation of a three-layer neural network for image classification on the CIFAR-10 dataset using pure NumPy. The model is trained to classify 10 categories of images from the CIFAR-10 dataset.

## Requirements

Ensure you have the following dependencies installed:

- numpy
- matplotlib
- torchvision

You can install the required packages using `pip`:

```bash
pip install numpy matplotlib torchvision
```

## Directory Structure

```
.
├── data_loader.py            # Data loading and preprocessing functions
├── activation_functions.py    # Activation functions and their backpropagation
├── three_layer_nn.py         # The 3-layer neural network implementation
├── training.py               # Training, hyperparameter search, and testing functions
├── visualization.py          # Visualization of weights and training curves
├── main.py                   # Main script to run the training and testing
├── README.md                 # This file
```

## How to Train the Model

Follow these steps to train the model:

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/cifar10-image-classification.git
cd cifar10-image-classification
```

### 2. Download the CIFAR-10 dataset:
The dataset will be automatically downloaded when you run the `main.py` script, but you must have an active internet connection.

### 3. Run the training script:
```bash
python main.py
```

This script will:
- Load the CIFAR-10 dataset.
- Preprocess the data (flatten images, normalize, and split into training, validation, and test sets).
- Train the neural network using grid search to tune hyperparameters such as learning rate, hidden layer size, and regularization strength.
- Save the best model parameters.

### Hyperparameters

The hyperparameters for the training process are defined in the `hyperparameter_search()` function in `training.py`. Currently, the grid search tries the following combinations:

- Learning rates: `1e-3`, `5e-4`
- Hidden layer sizes: `50`, `100`
- Regularization strengths: `0.0`, `1e-3`

You can modify these values in the code to experiment with different hyperparameters.

## How to Test the Model

Once training is complete, the model will automatically be evaluated on the test set.

### 1. Test the trained model:
After training, the model’s performance on the test set will be printed in the console.

### 2. Visualize the results:
The following visualizations will be generated:
- **Training Loss Curve**: The loss over training epochs.
- **Validation Accuracy Curve**: The accuracy over validation data for each epoch.
- **Hidden Layer Weights Visualization**: Visualization of the first hidden layer's weights (which represent learned features of CIFAR-10 images).

You can customize the visualization in `visualization.py` to add more plots or save the plots to image files.

## Model Weights Download

Once you have completed training, you can save the best model weights. These weights are saved in the `best_model.pkl` file.

To use a pre-trained model, you can load the saved weights using `pickle`:

```python
import pickle

with open('best_model.pkl', 'rb') as f:
    model_params = pickle.load(f)
```

### 3. Download Pre-trained Model (Optional)

For your convenience, you can download my pre-trained model from Google Drive.

- [Download model weights from Google Drive](https://drive.google.com/file/d/14kuJ0wIyZffxzAirXozzlna31PFh9lOr/view?usp=sharing)

## Results

The model achieved the following performance on the test set:

- **Test Accuracy**: 0.4486

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
