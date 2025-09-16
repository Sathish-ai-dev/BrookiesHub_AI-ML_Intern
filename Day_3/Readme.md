# Image Classification with CNN (CIFAR-10)

This project builds a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset** into 10 categories.  
We use **TensorFlow/Keras** for deep learning and visualize model performance with accuracy/loss plots and a confusion matrix.

---

## 🎯 Objective
- Train a CNN to classify 32x32 RGB images from CIFAR-10.  
- Use **Conv2D, MaxPooling2D, Flatten, Dense** layers.  
- Apply **data augmentation** for better generalization.  
- Evaluate the model with accuracy/loss graphs and confusion matrix.  

---

## 📂 Dataset
The **CIFAR-10 dataset** contains **60,000 images** (32x32 pixels, RGB) in 10 classes:

- **airplane**, **automobile**, **bird**, **cat**, **deer**  
- **dog**, **frog**, **horse**, **ship**, **truck**

- Training set: 50,000 images  
- Test set: 10,000 images  

It can be loaded directly from Keras:

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
