import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df=pd.read_csv("health.csv")
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())

num_cols=df.select_dtypes(include=['int64','float64']).columns
imputer=SimpleImputer(strategy='mean')
df[num_cols]=imputer.fit_transform(df[num_cols])

cat_cols=df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0],inplace=True)
le=LabelEncoder()
for col in cat_cols:
    df[col]=le.fit_transform(df[col])

target_col="Outcome"
X=df.drop(target_col,axis=1)
y=df[target_col]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


df.hist(figsize=(8,6))
plt.title("histogram")
plt.show()

sns.heatmap(df.corr(),annot=True)
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x=df.columns[0],y=df.columns[1],data=df)
plt.show()

t_model=DecisionTreeClassifier()
t_model.fit(X_train,y_train)
yt_pred=t_model.predict(X_test)

print("accuracy:",accuracy_score(y_test,yt_pred))
print("classification",classification_report(y_test,yt_pred))
print("confusion matrix",confusion_matrix(y_test,yt_pred))


l_model=LogisticRegression(max_iter=1000)
l_model.fit(X_train,y_train)
lt_pred=l_model.predict(X_test)

print("accuracy:",accuracy_score(y_test,lt_pred))
print("classification",classification_report(y_test,lt_pred))
print("confusion matrix",confusion_matrix(y_test,lt_pred))