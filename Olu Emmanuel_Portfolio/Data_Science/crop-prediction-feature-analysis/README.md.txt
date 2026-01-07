# Crop Prediction Feature Analysis 🌱

## Overview
This project identifies the **most predictive soil feature** for crop type prediction using Python and logistic regression.  
Understanding which soil characteristics impact crop selection can help farmers make **better planting decisions**.

---

## Dataset
- File: `soil_measures.csv`
- Features:
  - `N` (Nitrogen content)
  - `P` (Phosphorus content)
  - `K` (Potassium content)
  - `ph` (Soil pH)
- Target:
  - `crop` (Type of crop)

---

## Approach
1. **Data exploration**  
   Checked for missing values and data types, explored unique crop types.
2. **Data preparation**  
   Split the dataset into **train (70%)** and **test (30%)** sets.
3. **Modeling**
   - Trained a **Logistic Regression** model individually for each feature.
   - Evaluated performance using **weighted F1-score** to account for class imbalance.
4. **Feature scoring**
   - Stored F1-scores for each feature.
   - Selected the **best predictive feature** based on the highest score.

---

## Results

| Feature | F1-score |
|---------|----------|
| N       | 0.09   |
| P       | 0.11    |
| K       | 0.24     |
| ph      | 0.06     |

**Best predictive feature:** `k` (F1-score: 0.24) ✅

> This feature has the strongest impact on predicting crop type.

---

## Tools & Libraries
- Python 🐍
- pandas
- scikit-learn
- metrics: F1-score
- Jupyter Notebook (optional)

---