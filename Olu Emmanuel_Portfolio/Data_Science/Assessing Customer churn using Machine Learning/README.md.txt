# 📊 Telecom Customer Churn Prediction

This project builds and compares machine learning models to predict **customer churn** in a telecom company using demographic and usage data.  
The goal is to identify customers who are likely to leave, enabling proactive retention strategies.

---

## 🚀 Project Overview

Customer churn is a major challenge in the telecom industry. In this project, we:

- Merge customer demographic and usage datasets
- Perform data preprocessing and feature engineering
- Encode categorical variables and scale numerical features
- Train and evaluate multiple machine learning models
- Compare model performance using accuracy

---

## 📁 Dataset Description

The project uses two CSV files:

### 1. `telecom_demographics.csv`
Contains customer-level demographic information such as:
- Gender
- Senior citizen status
- Contract type
- Other customer attributes

### 2. `telecom_usage.csv`
Contains service usage information such as:
- Call duration
- Internet usage
- Billing-related variables

Both datasets are merged using a common key:  
**`customer_id`**

---

## ⚙️ Technologies & Libraries Used

- **Python**
- **Pandas & NumPy** – data manipulation
- **Scikit-learn** – modeling and evaluation

Models used:
- Logistic Regression
- Random Forest Classifier

---

## 🔄 Workflow

1. **Data Loading & Merging**
   - Read CSV files
   - Merge on `customer_id`

2. **Exploratory Step**
   - Calculate churn proportion

3. **Preprocessing**
   - Separate features (`X`) and target (`y`)
   - One-hot encode categorical variables
   - Standardize numerical features
   - Train-test split with stratification

4. **Model Training**
   - Train Logistic Regression
   - Train Random Forest Classifier

5. **Evaluation**
   - Compare models using accuracy score
   - Identify the better-performing model

---

## 📈 Model Evaluation

The models are evaluated using **accuracy** on the test dataset.

Example output:
```text
Logistic Regression Accuracy: 0.0.7938
Random Forest Accuracy: 0.0.7977
Higher Accuracy Model: Random Forest
