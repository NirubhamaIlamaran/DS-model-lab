import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

df=pd.read_csv("health.csv")

print(df.info())
print(df.head())
print(df.describe())
print("null values",df.isnull().sum())

num_cols=df.select_dtypes(include=['int64','float64']).columns
imputer=SimpleImputer(strategy='mean')
df[num_cols]=imputer.fit_transform(df[num_cols])

cat_cols=df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0],inplace=True)
le=LabelEncoder()
for col in cat_cols:
    df[col]=le.fit_transform(df[col])
df.hist(figsize=(8,6))
plt.title("histogram")
plt.show()
sns.heatmap(df.corr(),annot=True)
plt.title("heatmap")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(9,6))
sns.scatterplot(x=df.columns[0],y=df.columns[1],data=df)
plt.show()