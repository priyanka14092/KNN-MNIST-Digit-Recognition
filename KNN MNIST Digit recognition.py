#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dfx=pd.read_csv('xdata.csv')
dfy=pd.read_csv('ydata.csv')


# In[4]:


X=dfx.values
Y=dfy.values

X=X[:,1:]
Y=Y[:,1:].reshape((-1,))

print(X)
print(X.shape)
print(Y.shape)


# In[5]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[6]:


query_x=np.array([2,3])
plt.scatter(X[:,0],X[:,1],c=Y)
plt.scatter(query_x[0],query_x[1],color='red')
plt.show()


# In[15]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5):
    
    vals=[]
    
    m=X.shape[0]
    
    for i in range(m):
        d=dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    vals=sorted(vals)
    vals=vals[:k]
        
    vals=np.array(vals)
    
    #print(vals)
    
    new_vals=np.unique(vals[:,1],return_counts=True)
    print(new_vals)
    
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    
    return pred


# In[16]:


knn(X,Y,query_x)


# #MNIST DataSets

# In[17]:


df=pd.read_csv('train.csv')
print(df.shape)


# In[18]:


print(df.columns)


# In[19]:


df.head()


# In[20]:


#Create Numpy Array
data=df.values
print(data.shape)
print(type(data))


# In[21]:


X=data[:,1:]
Y=data[:,0]

print(X.shape,Y.shape)


# In[22]:


split=int(0.8*X.shape[0])
print(split)


# In[23]:


X_train=X[:split,:]
Y_train=Y[:split]
X_test=X[split:,:]
Y_test=Y[split:]

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[26]:


def drawImg(sample):
    img=sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()


# In[27]:


drawImg(X_train[3])
print(Y_train[3])


# #Making Prediction

# In[28]:


pred=knn(X_train,Y_train,X_test[0])
print(pred)


# In[35]:


drawImg(X_test[52])
print(Y_test[52])

