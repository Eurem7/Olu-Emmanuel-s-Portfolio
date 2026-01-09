<div align="center">

# 💳 Credit Card Fraud Detection System 🔍


## 🎯 Overview

</div>

<table>
<tr>
<td width="50%">

### 🛡️ Protection Metrics

- **98.67% Precision** - Virtually no false alarms
- **62.36% Recall** - Catches most fraud
- **99.80% Accuracy** - Reliable classification
- **0.20% FPR** - Minimal customer disruption

</td>
<td width="50%">

### 💼 Business Impact

- Protects financial assets effectively
- Reduces operational review costs
- Maintains customer trust
- Balances security with user experience

</td>
</tr>
</table>

---

## 📊 Results

<div align="center">

### Model Performance Comparison

<table>
<thead>
<tr>
<th>Metric</th>
<th>🏆 Random Forest</th>
<th>📈 Logistic Regression</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Precision</strong></td>
<td><span style="color: green;">✅ 98.67%</span></td>
<td><span style="color: red;">❌ 6.72%</span></td>
</tr>
<tr>
<td><strong>Recall</strong></td>
<td><span style="color: green;">✅ 62.36%</span></td>
<td><span style="color: orange;">⚠️ 71.91%</span></td>
</tr>
<tr>
<td><strong>F1-Score</strong></td>
<td><span style="color: green;">✅ 76.42%</span></td>
<td><span style="color: red;">❌ 12.29%</span></td>
</tr>
<tr>
<td><strong>ROC-AUC</strong></td>
<td><span style="color: green;">✅ 0.9778</span></td>
<td><span style="color: orange;">⚠️ 0.8341</span></td>
</tr>
<tr>
<td><strong>Accuracy</strong></td>
<td><span style="color: green;">✅ 99.80%</span></td>
<td><span style="color: orange;">⚠️ 94.62%</span></td>
</tr>
<tr>
<td><strong>False Positive Rate</strong></td>
<td><span style="color: green;">✅ 0.20%</span></td>
<td><span style="color: red;">❌ 5.26%</span></td>
</tr>
</tbody>
</table>

</div>

### 🎯 Confusion Matrix (Random Forest)

<div align="center">

```
                    Predicted
                Non-Fraud    Fraud
Actual  Non-Fraud  67,563       3
        Fraud         134     222
```

</div>

<details>
<summary><b>📈 Click to see detailed classification report</b></summary>

```
              precision    recall  f1-score   support

           0     0.9980    1.0000    0.9990     67566
           1     0.9867    0.6236    0.7642       356

    accuracy                         0.9980     67922
   macro avg     0.9923    0.8118    0.8816     67922
weighted avg     0.9980    0.9980    0.9978     67922
```

</details>

<br>

<div align="center">

**Real-world Impact:**

| Metric | Value | Description |
|--------|-------|-------------|
| ✅ Frauds Detected | 222 / 356 | Successfully caught 62.36% of fraud |
| ⚠️ Frauds Missed | 134 / 356 | Acceptable trade-off for precision |
| 🎯 Legitimate Flagged | 3 / 67,566 | Only 0.004% false positives |

</div>

---

## 🚀 Quick Start

<table>
<tr>
<td width="50%">

### Prerequisites

```bash
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

</td>
<td width="50%">

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

# Install dependencies
pip install -r requirements.txt
```

</td>
</tr>
</table>

### 💻 Usage

```python
# Import the fraud detection system
from fraud_detection import train_model, predict_fraud

# Train the model
model = train_model('credit_card_fraud.csv')

# Make predictions
predictions = predict_fraud(model, new_transactions)

# Output: Array of 0s (legitimate) and 1s (fraud)
```

<details>
<summary><b>🔧 Advanced Configuration</b></summary>

```python
# Custom model parameters
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    max_depth=20,
    min_samples_split=10,
    random_state=42
)

# Train with custom parameters
model.fit(X_train, y_train)
```

</details>

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── 📄 fraud_detection.py          # Main implementation
├── 📊 credit_card_fraud.csv       # Dataset (not in repo)
├── 📋 requirements.txt            # Dependencies
├── 📖 README.md                   # This file
├── 📜 LICENSE                     # MIT License
│
├── 📁 notebooks/
│   └── exploratory_analysis.ipynb # Data exploration
│
├── 📁 models/
│   └── saved_models/              # Trained model files
│
└── 📁 data/
    └── processed/                 # Preprocessed data
```

---

## 🔧 Technical Details

<div align="center">

### Architecture Overview

</div>

```mermaid
graph LR
    A[Raw Data] --> B[Feature Engineering]
    B --> C[Preprocessing]
    C --> D[Train/Test Split]
    D --> E[Model Training]
    E --> F[Random Forest]
    E --> G[Logistic Regression]
    F --> H[Evaluation]
    G --> H
    H --> I[Best Model Selection]
```

### 🎨 Feature Engineering

<table>
<tr>
<td width="33%">

**Temporal Features**
- Transaction hour
- Day of week
- Age calculation

</td>
<td width="33%">

**Categorical Encoding**
- Merchant encoding
- Category mapping
- Location encoding

</td>
<td width="33%">

**Numerical Processing**
- Amount normalization
- Distance calculations
- Statistical aggregations

</td>
</tr>
</table>

### 📊 Dataset Features

| Category | Features | Description |
|----------|----------|-------------|
| **Temporal** | `trans_date_trans_time`, `trans_hour` | When transaction occurred |
| **Personal** | `dob`, `age`, `job` | Customer demographics |
| **Transaction** | `merchant`, `category`, `amt` | Transaction details |
| **Location** | `city`, `state`, `lat`, `long` | Geographic information |
| **Target** | `is_fraud` | 0 = legitimate, 1 = fraud |

### 🔄 Preprocessing Pipeline

<div align="center">

```python
# 1. Feature Engineering
ccf['age'] = (ccf['trans_date_trans_time'] - ccf['dob']).dt.days // 365
ccf['trans_hour'] = ccf['trans_date_trans_time'].dt.hour

# 2. Categorical Encoding
cat_cols = ['merchant', 'category', 'city', 'state', 'job']
for col in cat_cols:
    ccf[col] = LabelEncoder().fit_transform(ccf[col])

# 3. Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Missing Value Imputation
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# 5. Feature Scaling (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

</div>

---

## 🤖 Models

<table>
<tr>
<td width="50%">

### 🏆 Random Forest (Primary)

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
```

**Why it wins:**
- ✅ Handles non-linear patterns
- ✅ Ensemble of 100 trees
- ✅ Robust to outliers
- ✅ Automatic feature selection

</td>
<td width="50%">

### 📈 Logistic Regression (Baseline)

```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
```

**Characteristics:**
- ⚠️ Linear decision boundary
- ⚠️ High recall, low precision
- ⚠️ Too many false positives
- ✅ Fast training/inference

</td>
</tr>
</table>

---

## 🎓 Key Features

<div align="center">

<table>
<tr>
<td align="center" width="25%">
<h3>⚙️</h3>
<h4>Automated Pipeline</h4>
Feature engineering and preprocessing
</td>
<td align="center" width="25%">
<h3>🔄</h3>
<h4>Class Balancing</h4>
Weighted loss for imbalanced data
</td>
<td align="center" width="25%">
<h3>📊</h3>
<h4>Rich Metrics</h4>
Comprehensive evaluation suite
</td>
<td align="center" width="25%">
<h3>🎯</h3>
<h4>Production Ready</h4>
Robust error handling
</td>
</tr>
</table>

</div>

---

