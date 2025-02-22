# MNIST ResNet Training

📌 **Project Overview:**

This project trains a **ResNet18** model on the **MNIST dataset** (handwritten digits 0-9) using **PyTorch**. The dataset is preprocessed, the model is fine-tuned, and its performance is evaluated.

## 📝 Data Preprocessing:
- Loads the **MNIST** dataset (handwritten digits 0-9).
- Converts grayscale images to **RGB** (3 channels) for compatibility with CNN models.
- Resizes images to **224x224** to match **ResNet18** input requirements.

## 🏗️ Model Selection & Modification:
- Uses a **pre-trained ResNet18** model from **PyTorch**.
- Replaces the final **fully connected layer** to classify 10 digits (0-9).

## 🎯 Training Process:
- Uses **CrossEntropyLoss** as the loss function.
- Optimizes the model using the **Adam optimizer**.
- Trains the model for **5 epochs** with a batch size of **32**.

## 📊 Model Evaluation:
- Evaluates the model on the **test dataset**.
- Computes **accuracy** based on correct predictions.

## 🚀 Results & Performance:
- Displays **loss per epoch** during training.
- Computes **final accuracy** on test data.

---

### 🛠️ How to Run:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
