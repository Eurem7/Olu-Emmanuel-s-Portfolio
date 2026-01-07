🎬 Movie Rental Duration Prediction

📌 Project Overview

This project focuses on predicting movie rental duration (in days) using customer rental information and movie features.
The goal is to apply regression modeling, feature engineering, and model selection techniques to identify the model that best predicts how long a movie is rented.

This project demonstrates:

Practical feature engineering from raw date fields

Use of multiple regression models

Hyperparameter tuning with cross-validation

Model comparison using Mean Squared Error (MSE)

🎯 Problem Statement

Given historical movie rental data, can we accurately predict how many days a movie will be rented?

Accurate predictions can help:

Improve inventory planning

Optimize pricing strategies

Enhance customer behavior analysis

📂 Dataset Description

The dataset (rental_info.csv) contains information about movie rentals, including:

Rental and return dates

Movie features

Rental attributes and customer-related variables

Target Variable

rental_length_days – number of days between rental and return dates

🛠️ Feature Engineering

Several important features were engineered to improve model performance:

Rental Duration

rental_length_days = return_date - rental_date


Binary Features from Text

deleted_scenes: 1 if Deleted Scenes is included

behind_the_scenes: 1 if Behind the Scenes is included

Dropped Columns

Raw date columns

Intermediate time delta columns

Text fields not directly usable by models

🤖 Models Trained

Three regression models were implemented and compared:

Linear Regression

Baseline model

Simple and interpretable

Decision Tree Regressor

Captures non-linear relationships

Tuned using RandomizedSearchCV

Random Forest Regressor

Ensemble model for improved generalization

Tuned using RandomizedSearchCV with cross-validation

🔍 Hyperparameter Tuning

Method: RandomizedSearchCV

Cross-Validation: 5-fold KFold

Scoring Metric: Negative Mean Squared Error

Example parameters tuned:

max_depth

min_samples_split

min_samples_leaf

n_estimators (Random Forest)

📊 Model Evaluation

Models were evaluated on a held-out test set using Mean Squared Error (MSE).

Model	Mean Squared Error
Linear Regression	Evaluated
Decision Tree	Evaluated
Random Forest	Best Performance ✅

📌 Best Model: Random Forest Regressor

📈 Feature Importance Analysis

Lasso Regression was used to visualize feature importance

Coefficients were plotted to understand which features influence rental duration the most

This improves model interpretability and feature selection insight.


🧰 Tools & Technologies

Python

pandas, NumPy

scikit-learn

matplotlib

Jupyter Notebook