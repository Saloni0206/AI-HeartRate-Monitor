# load_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load the dataset
# -----------------------------
data = pd.read_csv("heart_disease_uci.csv")  # Make sure this matches your CSV filename

# -----------------------------
# 2. Quick overview
# -----------------------------
print("First 5 rows of dataset:\n", data.head(), "\n")
print("Dataset info:\n")
data.info()
print("\nMissing values:\n", data.isnull().sum(), "\n")
print("Statistics:\n", data.describe(), "\n")

# -----------------------------
# 3. Target variable distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=data)
plt.title('Heart Disease Count (0 = No, 1 = Yes)')
plt.show()

# -----------------------------
# 4. Correlation heatmap
# -----------------------------
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# -----------------------------
# 5. Age distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(data['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# -----------------------------
# 6. Cholesterol vs. Heart Disease
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x='target', y='chol', data=data)
plt.title('Cholesterol Levels vs. Heart Disease')
plt.show()

# -----------------------------
# 7. Resting blood pressure vs. Heart Disease
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x='target', y='trestbps', data=data)
plt.title('Resting Blood Pressure vs. Heart Disease')
plt.show()

# -----------------------------
# 8. Scatter plot: Age vs. Max Heart Rate
# -----------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='thalach', hue='target', data=data)
plt.title('Age vs. Max Heart Rate')
plt.show()

# -----------------------------
# 9. Pairplot (optional, can be slow)
# -----------------------------
# sns.pairplot(data, hue='target')
# plt.show()

