#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris
import pandas as pd
iris_dataset = load_iris()


# In[3]:


print(iris_dataset.keys())


# In[4]:


print(iris_dataset['feature_names'])


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)


# In[7]:


iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3) 


# In[8]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# In[12]:


print("Accuracy: {}%".format((knn.score(X_test, y_test))*100))


# In[ ]:




