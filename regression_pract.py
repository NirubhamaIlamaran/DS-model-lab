import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


df=pd.read_csv('health.csv')
print(df.info())
print(df.head())
print(df.describe())
print("Null values",df.isnull().sum())


num_col= df.select_dtypes(include=['int64','float64']).columns
impute=SimpleImputer(strategy='mean')
df[num_col]=impute.fit_transform(df[num_col])


cat_col=df.select_dtypes(include='object').columns

for col in cat_col:
    df[col].fillna(df[col].mode()[0],inplace=True)
le=LabelEncoder()
for col in cat_col:
    le.fit_transform(df[col])


target_col='Health_Score'
X=df.drop(target_col,axis=1)
y=df[target_col]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

df.hist(figsize=(8,6))
plt.title('histogram')
plt.show()

sns.heatmap(df.corr(),annot=True)
plt.title('heatmap')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()

sns.scatterplot(x=df.columns[0],y=df.columns[1],data=df)
plt.show()

dt_model=DecisionTreeRegressor()
dt_model.fit(X_train,y_train)
yt_pred=dt_model.predict(X_test)

print("MSE:",np.sqrt(mean_squared_error(y_test,yt_pred)))
print("r2_score",r2_score(y_test,yt_pred))



lt_model=LinearRegression()
lt_model.fit(X_train,y_train)
lt_pred=lt_model.predict(X_test)

print("MSE:",np.sqrt(mean_squared_error(y_test,yt_pred)))
print("r2_score",r2_score(y_test,yt_pred))




