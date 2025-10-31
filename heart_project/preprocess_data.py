import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
data = pd.read_csv("heart_disease_uci.csv")

# -----------------------------
# Step 1: Handle missing values
# -----------------------------
# Numerical columns
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    data[col].fillna(data[col].mean(), inplace=True)

# Categorical columns
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# -----------------------------
# Step 2: Encode categorical columns
# -----------------------------
le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

# -----------------------------
# Step 3: Scale numerical features
# -----------------------------
scaler = MinMaxScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# -----------------------------
# Step 4: Split features and target
# -----------------------------
# Assuming 'num' column is the target (0 = no disease, 1 = disease)
X = data.drop('num', axis=1)
y = data['num']

print("Preprocessing complete!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
