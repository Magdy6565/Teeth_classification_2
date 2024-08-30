# Teeth_classification_2
Here's a `README.md` file for your project:

```markdown
# Teeth Disease Classification Project

This project focuses on classifying images of teeth into one of seven possible disease categories using a deep learning model based on EfficientNetB0. The model is trained using TensorFlow and Keras, and a Streamlit application is provided to allow users to upload images and classify them.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Training](#model-training)
- [Streamlit Application](#streamlit-application)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Project Structure

```
├── data/
│   ├── Training/
│   ├── Validation/
│   ├── Testing/
├── models/
│   └── best_model.h5
├── streamlit_app.py
├── train.py
├── README.md
```

- **data/**: Directory containing the training, validation, and testing datasets.
- **models/**: Directory to store trained models.
- **streamlit_app.py**: Streamlit application for image classification.
- **train.py**: Script to train the EfficientNetB0 model.
- **README.md**: Project documentation.

## Requirements

To run this project, you need the following libraries:

- TensorFlow
- Streamlit
- Pillow

You can install the required packages with the following command:

```bash
pip install tensorflow streamlit pillow
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/teeth-disease-classification.git
   cd teeth-disease-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Place your dataset in the `data/` directory with subdirectories for `Training`, `Validation`, and `Testing`.

## Model Training

The model is based on EfficientNetB0, pre-trained on ImageNet, with a custom classification head for 7 disease categories. The training script (`train.py`) handles data loading, augmentation, and training.

### To train the model:

```bash
python train.py
```

### Model Architecture:

- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Head**:
  - Global Average Pooling
  - Dense Layer (256 units, ReLU activation)
  - Dropout (0.5)
  - Dense Layer (7 units, Softmax activation)

### Training Details:

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

The best model is saved in the `models/` directory as `best_model.h5`.

## Streamlit Application

The `streamlit_app.py` allows users to upload an image of teeth and classify it into one of the 7 disease categories.

### To run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Usage

1. **Upload an Image**: Use the Streamlit app to upload an image of teeth.
2. **Classify**: The model will predict the disease category.

## Acknowledgments

- The EfficientNetB0 model is pre-trained on ImageNet.
- Special thanks to the TensorFlow and Keras communities for providing the tools to build and deploy deep learning models.

```
