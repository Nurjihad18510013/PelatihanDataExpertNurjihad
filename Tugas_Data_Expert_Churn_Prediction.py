#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library yang dibutuhkan

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from pathlib import Path


# In[2]:


#import dataset
df_load = pd.read_csv("C:/Data_Expert/Tugas/churn_telco_final.csv")

#tampilkan bentuk dari dataset
print(df_load.shape)

#tampilkan 5 data teratas
print(df_load.head())

#tampilkan jumlah ID yang unik
print(df_load.customerID.nunique())


# In[3]:


#import matplotlib dan seaborn
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from matplotlib import pyplot as plt
import numpy as np
#Code
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
labels = ['Yes', 'No']
churn = df_load.Churn.value_counts()
ax.pie(churn, labels=labels, autopct='%.0f%%')
plt.show()


# In[5]:


from matplotlib import pyplot as plt
import numpy as np
#chart
numerical_features = ['MonthlyCharges', 'TotalCharges', 'tenure']
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
#Code
df_load[df_load.Churn == 'No'][numerical_features].hist(bins=20, color='blue', alpha=0.5, ax=ax)
df_load[df_load.Churn == 'Yes'][numerical_features].hist(bins=20, color='orange', alpha=0.5, ax=ax)
plt.show()


# In[6]:


from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style='darkgrid')

fig, ax = plt.subplots(3, 3, figsize=(14, 12))
sns.countplot(data=df_load, x='gender', hue='Churn', ax=ax[0][0])
sns.countplot(data=df_load, x='Partner', hue='Churn', ax=ax[0][1])
sns.countplot(data=df_load, x='SeniorCitizen', hue='Churn', ax=ax[0][2])
sns.countplot(data=df_load, x='PhoneService', hue='Churn', ax=ax[1][0])
sns.countplot(data=df_load, x='StreamingTV', hue='Churn', ax=ax[1][1])
sns.countplot(data=df_load, x='InternetService', hue='Churn', ax=ax[1][2])
sns.countplot(data=df_load, x='PaperlessBilling', hue='Churn', ax=ax[2][1])
plt.tight_layout()
plt.show()


# In[8]:


#Remove the unnecessary columns customerID & UpdateAt

cleaned_df = df_load.drop(['customerID','UpdatedAt'], axis=1)
print(cleaned_df.head())


# In[10]:


from sklearn.preprocessing import LabelEncoder
#Convert all the non-numeric columns to numerical data types
for column in cleaned_df.columns:
    if cleaned_df[column].dtype == np.number: continue
    #Perform encoding for each non-numeric column
    cleaned_df[column] = LabelEncoder().fit_transform(cleaned_df[column])
print(cleaned_df.describe())


# In[11]:


from sklearn.model_selection import train_test_split
# Predictor and Target
X = cleaned_df.drop('Churn', axis = 1)
y = cleaned_df['Churn']
# Spliting train and test
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
# print according
print('Jumlah baris dan kolom dari x_train adalah:', x_train.shape,',sedangkan Jumlah baris dan kolom dari y_train adalah:', y_train.shape)
print('Prosentase Churn di data Training adalah:')
print(y_train.value_counts(normalize=True))
print('Jumlah baris dan kolom dari x_test adalah:', x_test.shape,',sedangkan Jumlah baris dan kolom dari y_test adalah:', y_test.shape)
print('Prosentase Churn di data Testing adalah:')
print(y_test.value_counts(normalize=True))


# In[12]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression().fit(x_train,y_train)
print('Model Logistic Regression yang terbentuk adalah: \n', log_model)


# In[15]:


from sklearn.metrics import classification_report
#Predictor
y_train_pred = log_model.predict(x_train)
print('Model classification reportnya(LogisticRegression):')
print(classification_report(y_train, y_train_pred))


# In[ ]:




