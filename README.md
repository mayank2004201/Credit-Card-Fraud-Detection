# Credit Card Fraud Detection

## Overview
Credit card fraud is a major challenge in modern financial systems, causing billions of dollars in losses every year. Detecting fraudulent transactions quickly and accurately is essential for financial institutions.

This project builds a **machine learning model to detect fraudulent credit card transactions** using historical transaction data. The model learns patterns that distinguish legitimate transactions from fraudulent ones and helps identify suspicious activity.

---

## Problem Statement
Fraud detection datasets are typically **highly imbalanced**, where fraudulent transactions represent only a very small portion of the total data. This imbalance makes it difficult for traditional machine learning models to detect fraud effectively.

The goal of this project is to build a model that can **accurately detect fraudulent transactions while minimizing false positives**.

---

## Dataset
The dataset contains credit card transactions made by cardholders. It includes both legitimate and fraudulent transactions.

Key characteristics of the dataset:

- Highly imbalanced transaction classes
- Anonymized features for privacy
- Includes transaction time and transaction amount

Target variable:

- `0` → Legitimate transaction  
- `1` → Fraudulent transaction  

---

## Project Workflow

### 1. Data Exploration
Understanding the dataset using:
- Class distribution analysis
- Feature relationships
- Transaction amount patterns

### 2. Data Preprocessing
Key preprocessing steps include:
- Handling class imbalance
- Feature scaling
- Train-test split
- Data normalization

### 3. Feature Engineering
Relevant features are selected and prepared to improve model performance and prediction accuracy.

### 4. Model Training
Different machine learning models can be trained for fraud detection, such as:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

These models learn patterns that help distinguish fraudulent transactions from legitimate ones.

### 5. Model Evaluation
Since the dataset is imbalanced, accuracy alone is not a reliable metric. The following evaluation metrics are used:

- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Score

These metrics provide a better understanding of how well the model detects fraud.

---

## Results
The trained model is able to detect fraudulent transactions effectively while maintaining a balance between recall and precision. This ensures that fraudulent activities are identified without excessively flagging legitimate transactions.

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  

---

## How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/mayank2004201/Credit-Card-Fraud-Detection.git
```

### 2. Navigate to the project directory

```
cd Credit-Card-Fraud-Detection
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the notebook

Open the Jupyter notebook and execute the cells to train and evaluate the model.

---

## Applications

This type of fraud detection system can be used in:

- Banking systems
- Online payment platforms
- Financial fraud monitoring
- Risk management systems

---

## Future Improvements

- Deploy the model as a REST API
- Implement real-time fraud detection
- Use advanced models such as XGBoost or deep learning
- Build an automated ML pipeline

---

## Author

**Mayank Goel**
