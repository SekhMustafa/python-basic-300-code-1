import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

!pip install xgboost
df = pd.read_csv("bitcoin.csv")
df.head()

df.shape

df.describe()

df.info()

plt.figure(figsize = ( 15, 5))
plt.plot(df["Close"])
plt.title("Bitcoin close price.", fontsize = 15)
plt.ylabel("price in dollars .")
plt.show()

df[df["Close"] == df["Adj Close"]].shape, df.shape

df = df.drop(["Adj Close"], axis =1)


df.isnull().sum()

feature = ["Open", "High", "Low", "Close"]

plt.subplots(figsize = (20,10))
for i, col in enumerate(feature):
    plt.subplot(2, 2, i+1)
    sns.distplot(df[col])
plt.show()

plt.subplots(figsize = (20, 10))
for i, col in enumerate(feature):
    plt.subplot(2,2, i+1)
    sns.boxplot(df[col], orient = "h")
plt.show()

splitted = df["Date"].str.split("-", expand = True)
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

df['Date'] = pd.to_datetime(df['Date'])

df.head()

df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

df['is_quarter_end'] = np.where(df['month'] %3 == 0,1,0)
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming df is already defined
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Scaling the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and validation (test) sets
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.3, random_state=42)

# 'test_size=0.3' means 30% of the data will be used for testing, and 70% for training

models = [LogisticRegression(), SVC(kernel = 'poly', probability = True),
          XGBClassifier()]
for i in range(3):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')
    print('Training Accuracy :' , metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:, 1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
    print()
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid , cmap = 'Blues')
plt.show()


