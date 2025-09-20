import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from scipy import stats
import kagglehub

# ----------------------------
# 1. Download Dataset
# ----------------------------
DATASET_NAME = "valakhorasani/bank-transaction-dataset-for-fraud-detection"
dataset_path = kagglehub.dataset_download(DATASET_NAME)

# Load CSV (assume only one CSV)
csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
if len(csv_files) == 0:
    raise FileNotFoundError("No CSV file found in downloaded dataset.")
data = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
print("Dataset loaded. Shape:", data.shape)

# ----------------------------
# 2. Define numeric & categorical columns
# ----------------------------
numeric_cols = [
    "TransactionAmount",
    "AccountBalance",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
]

categorical_cols = [
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "Channel",
    "CustomerOccupation",
]

# ----------------------------
# 3. Data Exploration
# ----------------------------
# Histograms
data[numeric_cols].hist(bins=30, figsize=(12, 8))
plt.suptitle("Numeric Feature Distributions")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# ----------------------------
# 4. Detect Data Issues
# ----------------------------
print("\n--- Data Issues Detection ---")
# Missing Values
missing_values = data.isnull().sum()
if missing_values.sum() == 0:
    print("✅ No missing values detected.")
else:
    print("Missing Values per Column:")
    for col, count in missing_values.items():
        if count > 0:
            print(f"⚠ {col}: {count} missing values")

# Outliers (Z-score > 3)
z_scores = np.abs(stats.zscore(data[numeric_cols]))
outlier_counts = (z_scores > 3).sum(axis=0)
print("\nOutliers per Numeric Column (Z-score > 3):")
for col, count in zip(numeric_cols, outlier_counts):
    print(f"{col}: {count} outlier(s)")

# Duplicate rows
duplicates = data.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# ----------------------------
# 5. Data Cleaning & Transformation
# ----------------------------
print("\n--- Data Cleaning & Transformation ---")

# Handle missing values
for col in numeric_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())
for col in categorical_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])
print("✅ Missing values handled.")

# Remove outliers
initial_rows = data.shape[0]
data = data[(z_scores < 3).all(axis=1)]
removed_rows = initial_rows - data.shape[0]
print(f"✅ Removed {removed_rows} outlier rows.")

# Clean categorical columns
for col in categorical_cols:
    data[col] = data[col].astype(str).str.strip().str.lower()
print("✅ Categorical columns cleaned.")

# Scale numeric features
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
print("✅ Numeric features scaled.")

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
print("✅ Categorical features encoded.")

# ----------------------------
# 6. Create Synthetic IsFraud Column
# ----------------------------
print("\n--- Creating Synthetic IsFraud Column ---")
data["IsFraud"] = 0
data.loc[data["TransactionAmount"] > 10000, "IsFraud"] = 1
data.loc[data["LoginAttempts"] > 3, "IsFraud"] = 1
data.loc[data["AccountBalance"] < 0, "IsFraud"] = 1
print("✅ Synthetic IsFraud column created.")
print(data["IsFraud"].value_counts())

# ----------------------------
# 7. Random Forest Classifier
# ----------------------------
X = data.drop(
    columns=[
        "IsFraud",
        "TransactionID",
        "AccountID",
        "TransactionDate",
        "PreviousTransactionDate",
    ],
    errors="ignore",
)
y = data["IsFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ----------------------------
# 8. Model Evaluation
# ----------------------------
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 9. Feature Importance
# ----------------------------
importances = rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame(
    {"Feature": features, "Importance": importances}
).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Random Forest Feature Importance")
plt.show()

print("\n--- Script Completed ---")
