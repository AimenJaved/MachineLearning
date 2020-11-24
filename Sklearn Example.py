#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import datasets
# Get data or import Data
iris = datasets.load_iris()


# In[30]:


# Preprocess the Data


# In[3]:


# split it in features and labels/targets
# featurization
features = iris.data
labels = iris.target


# In[4]:


features


# In[5]:


labels


# In[6]:


from sklearn.model_selection import train_test_split
# split into traning and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


# In[7]:


X_train


# In[8]:


# Choose a Machine Learning model
from sklearn import svm


# In[9]:


# Train it
model = svm.SVC()
model.fit(X_train,y_train)


# In[10]:


# predict using the model
model.predict(X_test)


# In[11]:


y_test


# In[12]:


# Save the trained model for later use
import pickle
with open('irisclassifier.pkl', 'wb') as f:
    pickle.dump(model, f)

