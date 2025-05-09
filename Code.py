import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("LoanApprovalPrediction.csv")

data.head(5)

obj = (data.dtypes =='object')
print("categorical variables:", len(list(obj[obj].index)))

data.drop(['Loan_ID'], axis=1,inplace=True)

obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(18,36))
index = 1

for col in object_cols:
    y= data[col].value_counts()
    plt.subplot(11,4,index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index +=1

#import label Encoder

from sklearn import preprocessing

#Label_encoder object knows how
#to understand word labels.

label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

obj = (data.dtypes == 'object')
print('Catagorical variables:', len(list(obj[obj].index)))

plt.figure(figsize=(12,6))
sns.heatmap(data.corr(), cmap='BrBG', fmt= '.2f',
            linewidth =2, annot=True)

sns.catplot(x='Gender', y='Married',
            hue="Loan_Status",
            kind="bar",
            data=data)

for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())
data.isna().sum()

from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']
X.shape,Y.shape

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.4, random_state=1)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



