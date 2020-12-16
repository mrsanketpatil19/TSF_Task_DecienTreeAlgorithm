
# ------------------ Importing Required Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# ----------------- Reading The Dataset

df = pd.read_csv("F:/TSF/Iris.csv")

df.info()
df.describe()
df.shape

# ------------ Checking Unique values

for val in df:
    if df[val].unique().shape[0]==df.shape[0]:
        print('Remove Column',val)
    print(val, ' ', df[val].unique().shape)

df = df.drop('Id',axis=1)

# ------------ Checking null values

df.isnull().any()

# ------------ Checking datatypes

df.dtypes

# ------------ EDA

df["types"] = df["Species"]

%matplotlib inline

sns.set(font_scale=1.5)
sns.pairplot(df,hue="types",size=3);
plt.show()

# -------------- Corrplot and Heatmap

corrmat = df.corr()
corrmat
fig,ax = plt.subplots()
fig.set_size_inches(34,34)
sns.heatmap(corrmat,annot = True)

# ------------ Seperating Dataset

Independent_var = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

Dependent_var = df['Species']

x_train,x_test,y_train,y_test = train_test_split(Independent_var,Dependent_var, random_state=10, test_size=0.3)

# ----------- Implementing Model

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

predictions = dt.predict(x_test)

dt.score(x_test, y_test)

print(classification_report(y_test, predictions, target_names=["Iris-setosa","Iris-versicolor","Iris-virginica"]))


