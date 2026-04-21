# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("health.csv")

# ==============================
# 3. Basic EDA
# ==============================
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# ==============================
# 4. Preprocessing
# ==============================

# Handle missing values
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])

# Encode categorical data
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ==============================
# 5. Features & Target
# ==============================
target_col = 'Health'   # change if needed

X = df.drop(target_col, axis=1)
y = df[target_col]

# ==============================
# 6. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ==============================
# 7. Feature Scaling (optional for tree)
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 8. Visualization
# ==============================

# Histogram
df.hist(figsize=(8,6))
plt.show()

# Heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()

# ==============================
# 9. Decision Tree Model
# ==============================
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 10. Final Result
# ==============================
print("\nFinal Accuracy:", accuracy_score(y_test, y_pred))