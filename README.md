# 🏦 Bank Customer Churn Prediction

A Machine Learning project that predicts whether a bank customer is likely to **churn (leave the bank)** based on demographic, financial, and account-related information.

The project implements a complete machine learning workflow including **data preprocessing, categorical encoding, feature scaling, model training, evaluation, model serialization, and customer-level prediction** using Python and Scikit-learn.

---

## 📌 Project Overview

Customer churn is an important challenge in the banking industry. Identifying customers who are likely to leave can help organizations understand customer behavior and develop appropriate retention strategies.

This project uses historical bank customer data to train a **Logistic Regression classification model** that predicts whether a customer is likely to churn.

The trained model can then be used to make predictions for new customer records.

---

## 🎯 Objectives

* Analyze bank customer information
* Prepare and preprocess the dataset
* Convert categorical features into numerical values
* Scale numerical features using StandardScaler
* Train a Logistic Regression classification model
* Evaluate model performance
* Save the trained model and scaler
* Predict churn for individual customers

---

## 🧠 Machine Learning Workflow

```text
Bank Customer Dataset
        │
        ▼
Data Loading
        │
        ▼
Data Cleaning
        │
        ▼
Remove Unnecessary Columns
        │
        ▼
Categorical Encoding
        │
        ▼
Train / Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Logistic Regression
        │
        ▼
Model Evaluation
        │
        ▼
Save Model + Scaler
        │
        ▼
Customer Churn Prediction
```

---

## 📊 Dataset

The project uses the **Churn Modelling** dataset containing information about bank customers.

Important features include:

| Feature         | Description                              |
| --------------- | ---------------------------------------- |
| CreditScore     | Customer's credit score                  |
| Geography       | Customer's country/region                |
| Gender          | Customer gender                          |
| Age             | Customer age                             |
| Tenure          | Number of years with the bank            |
| Balance         | Account balance                          |
| NumOfProducts   | Number of bank products used             |
| HasCrCard       | Whether the customer has a credit card   |
| IsActiveMember  | Whether the customer is an active member |
| EstimatedSalary | Estimated customer salary                |
| Exited          | Target variable indicating churn         |

The columns `RowNumber`, `CustomerId`, and `Surname` are removed during preprocessing because they are not used as predictive features.

---

## 🤖 Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as the classification algorithm.

Logistic Regression is suitable for binary classification problems where the target has two possible outcomes:

```text
0 → Customer is likely to stay
1 → Customer may churn
```

The model is trained using the preprocessed customer features.

---

## 🔄 Data Preprocessing

The following preprocessing steps are performed:

### 1. Remove unnecessary columns

```python
data = data.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1
)
```

### 2. Separate features and target

```python
X = data.drop("Exited", axis=1)
y = data["Exited"]
```

### 3. Encode categorical variables

The categorical columns `Geography` and `Gender` are converted into numerical features using one-hot encoding.

```python
X = pd.get_dummies(
    X,
    columns=["Geography", "Gender"],
    drop_first=True
)
```

### 4. Split the dataset

The dataset is divided into training and testing sets using an **80/20 split**.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 5. Feature scaling

`StandardScaler` is used to standardize the input features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 📈 Model Training

The Logistic Regression model is trained with:

```python
model = LogisticRegression(max_iter=1000)

model.fit(
    X_train_scaled,
    y_train
)
```

The trained model is evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

---

## 💾 Model Saving

After training, both the trained model and scaler are saved together using Joblib.

```python
model_data = {
    "model": model,
    "scaler": scaler
}

joblib.dump(
    model_data,
    "model.pkl"
)
```

This allows the saved model to be reused without training it again.

---

## 🔮 Prediction

The `predict.py` script allows users to enter customer information through the terminal.

Example inputs include:

```text
Credit Score
Age
Tenure
Balance
Number of Products
Credit Card Status
Active Member Status
Estimated Salary
Geography
Gender
```

The input is transformed using the same preprocessing approach used during training.

The saved model then produces the prediction.

### Example output

```text
Customer may churn.
```

or

```text
Customer is likely to stay.
```

---

## 📁 Project Structure

```text
Bank_customer_churn_predection/
│
├── data/
│   └── Churn_Modelling.csv
│
├── model.pkl
│
├── train.py
│
├── predict.py
│
├── .gitignore
│
└── README.md
```

---

## 🛠️ Technologies Used

| Technology          | Purpose                        |
| ------------------- | ------------------------------ |
| Python              | Programming language           |
| Pandas              | Data loading and preprocessing |
| Scikit-learn        | Machine Learning               |
| Logistic Regression | Churn classification           |
| StandardScaler      | Feature scaling                |
| Joblib              | Model serialization            |
| CSV                 | Dataset format                 |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/gandiharshavardhan1604-ux/Bank_customer_churn_predection.git
```

### 2. Navigate to the project

```bash
cd Bank_customer_churn_predection
```

### 3. Install dependencies

```bash
pip install pandas scikit-learn joblib
```

---

## ▶️ Train the Model

Run:

```bash
python train.py
```

The script will:

1. Load the dataset
2. Display dataset information
3. Remove unnecessary columns
4. Encode categorical features
5. Split the dataset
6. Scale the features
7. Train the Logistic Regression model
8. Evaluate the model
9. Save the model as `model.pkl`

---

## 🔍 Make a Prediction

After the model has been trained, run:

```bash
python predict.py
```

You will be prompted to enter customer information.

Example:

```text
Enter Credit Score: 650
Enter Age: 35
Enter Tenure: 5
Enter Balance: 50000
Enter Number of Products: 2
Has Credit Card? (1=Yes, 0=No): 1
Is Active Member? (1=Yes, 0=No): 1
Enter Estimated Salary: 75000
Enter Geography (France/Spain/Germany): France
Enter Gender (Male/Female): Male
```

The model will then display the predicted churn status.

---

## 📊 Evaluation

The training script evaluates the model using:

### Accuracy

Measures the overall percentage of correct predictions.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

The evaluation results are printed directly when `train.py` is executed.

---

## 💡 Key Learning Outcomes

This project demonstrates practical understanding of:

* Data preprocessing
* Exploratory dataset inspection
* Feature-target separation
* One-hot encoding
* Train-test splitting
* Feature standardization
* Binary classification
* Logistic Regression
* Model evaluation
* Confusion matrices
* Classification reports
* Model serialization
* Making predictions using a saved ML model

---

## 🚀 Future Enhancements

Possible improvements include:

* Add additional machine learning algorithms such as Random Forest, XGBoost, and SVM
* Compare multiple models using common evaluation metrics
* Add hyperparameter tuning
* Add cross-validation
* Build an interactive Streamlit web application
* Display churn probability instead of only the predicted class
* Add visualizations for customer churn analysis
* Add feature importance and model explainability
* Deploy the prediction application online

---

## ⚠️ Disclaimer

This project is developed for **educational and portfolio purposes**. Model predictions are based on the dataset and trained model and should not be treated as financial advice or as a production banking decision system.

---

## 👨‍💻 Author

**Gandi Harsha Vardhan**

AI-focused Computer Science Undergraduate

GitHub:
https://github.com/gandiharshavardhan1604-ux

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
