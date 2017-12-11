import matplotlib.pyplot as plt
import matplotlib.figure as figure
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.gridspec as gridspec
from Optimization import Optimization

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')

# Seperating fraudulent and legitimate transactions for Time variable
dataset.iloc[:,0][(dataset.iloc[:,30]==1)].describe()
dataset.iloc[:,0][(dataset.iloc[:,30]==0)].describe()

# Seperating fraudulent and legitimate transactions for Time variable
fraud = dataset.iloc[:,0][(dataset.iloc[:,30]==1)]
legit = dataset.iloc[:,0][(dataset.iloc[:,30]==0)]

# Type of charge vs Times
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))
ax1.hist(fraud, 60)
ax2.hist(legit, 60)
ax1.set_title('Fraud')
ax2.set_title('Legit')
plt.xlabel('Time (Sec)')
plt.ylabel('Number of Transactions (over 60 sec)')
plt.show()

# Seperating fraudulent and legitimate transactions for Transaction Amount
dataset.iloc[:,29][(dataset.iloc[:,30]==1)].describe()
dataset.iloc[:,29][(dataset.iloc[:,30]==0)].describe()

# Seperating fraudulent and legitimate transactions for Transaction variable
fraud_amount = dataset.iloc[:,29][(dataset.iloc[:,30]==1)]
legit_amount = dataset.iloc[:,29][(dataset.iloc[:,30]==0)]

# Type of charge vs Amount
f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(12,6))
ax1.hist(fraud_amount, 40)
ax2.hist(legit_amount, 40)
ax1.set_title('Fraud')
ax2.set_title('Legit')
plt.xlabel('Transaction Amount ($)')
plt.ylabel('Number of Transactions (over 60 sec)')
plt.yscale('log')
plt.show()

# Type of charge vs Amount
f, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(12,8))
ax1.scatter(fraud, fraud_amount)
ax2.scatter(legit, legit_amount)
ax1.set_title('Fraud')
ax2.set_title('Legit')
plt.xlabel('Time Fraud Occured')
plt.ylabel('Transactions Amount ($)')
plt.show()

#Select only the anonymized features.
v_features = dataset.iloc[:,1:29].columns

plt.figure(figsize=(12,120))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(dataset[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(dataset[cn][dataset.Class == 1], bins=50)
    sns.distplot(dataset[cn][dataset.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()

# Fraud vs Legitimate Cases
count_classes = pd.value_counts(dataset['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

# Removing Time & Amount
X = dataset.iloc[:,1:29]
y = dataset.iloc[:, 30]

# Randomly choose an equal # of 'legit' cases as there are fraudulent cases
X_fraud = X[y==1]
X_legit = X[y==0]
legit_index = np.array(np.random.choice(X_legit.index, len(X_fraud), replace=False))

X_legit_undersample = X.iloc[legit_index,:]
y_legit_undersample = y.iloc[legit_index]

# Sample to be trained
X_undersample = np.concatenate([X_fraud, X_legit_undersample])
y_undersample = np.concatenate([y[y==1], y_legit_undersample])

# Undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size = 0.3, random_state = 0)

# Feature Scaling (MUST BE APPLIED IN DIMENSIONALITY REDUCTION)
sc = StandardScaler()
X_train = sc.fit_transform(X_train_undersample)
X_test = sc.transform(X_test_undersample)

# Apply PCA, K-NN, KFold and Keras Optimization Functions
opt = Optimization(X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample)
opt.pca_optimization()
opt.cls_optimization()
opt.kfold_optimization()
opt.keras_optimization(5, 30, 100, 500)
