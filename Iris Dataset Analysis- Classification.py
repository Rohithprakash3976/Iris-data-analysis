#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import modules


# In[2]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


##Loading the data set


# In[4]:


df = pd.read_csv('Iris.csv')
df.head()


# In[5]:


df = df.drop(columns= ['Id'])
df.head()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


#To display no.of samples on each class
df['Species'].value_counts()


# In[9]:


##Preprocessing the data


# In[10]:


#check for null values
df.isnull().sum()


# In[11]:


##Exploratory Data Analysis


# In[12]:


df['SepalLengthCm'].hist()


# In[13]:


df['SepalWidthCm'].hist()


# In[14]:


df['PetalLengthCm'].hist()


# In[15]:


df['PetalWidthCm'].hist()


# In[16]:


#scatter plot
colors = ['red', 'orange', 'blue']
species = ['Iris-setosa',  'Iris-versicolor', 'Iris-virginica']


# In[18]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal width")
plt.legend()


# In[19]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal width")
plt.legend()


# In[20]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[22]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal width")
plt.legend()


# In[23]:


##Correlation metrix


# In[24]:


df.corr()


# In[26]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax)


# In[27]:


##Label encoder


# In[28]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[29]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[30]:


##Model Training


# In[31]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30) 


# In[32]:


#Logistic Regression


# In[33]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[35]:


model.fit(X_train, y_train)


# In[36]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[37]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[39]:


model.fit(X_train, y_train)


# In[40]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[ ]:




