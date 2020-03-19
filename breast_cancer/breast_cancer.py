#!/usr/bin/env python
# coding: utf-8

# In[2]:



X, y = mglearn.datasets.make_forge()
get_ipython().run_line_magic('matplotlib', 'inline')
mglern.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend("Class 0", "Class 1", loc=4)
plt.xlabel("First label")
plt.ylabel("Second label")
print("Shape X: {}".format(X.shape))


# In[2]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())


# In[ ]:


print(cancer['DESCR'])


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], random_state=0)


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)


# In[13]:


print(knn.score(X_test, y_test))


# In[ ]:




