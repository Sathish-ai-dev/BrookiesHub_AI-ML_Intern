#  Titanic Survival Prediction with Logistic Regression

This project uses **Logistic Regression** to predict whether a passenger survived the Titanic disaster, based on features such as age, sex, passenger class, and fare.  
It is a classic **binary classification problem** and a great introduction to **machine learning with scikit-learn**.

---

##  Objective
- Predict survival (`0 = Died`, `1 = Survived`) using passenger data.  
- Preprocess the dataset by handling missing values and encoding categorical variables.  
- Train a **Logistic Regression classifier**.  
- Evaluate with accuracy, confusion matrix, classification report, and ROC curve.  

---

##  Dataset
The dataset used is the **Titanic dataset**:  
- Features: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`  
- Target: `survived` (0 = No, 1 = Yes)

# Deliverables

- Preprocessing: handled missing values, encoded categorical features.
- Model: Logistic Regression trained on Titanic dataset.
- Accuracy Report: printed classification report + accuracy.
- ROC Curve: plotted with AUC score.

# Typical accuracy: 
 - ~78–82% depending on preprocessing.
