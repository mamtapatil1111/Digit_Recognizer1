#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils


# In[2]:


train_data=np.genfromtxt('train.csv',delimiter=',',dtype='int',skip_header=1)
test_data=np.genfromtxt('test.csv',delimiter=',',dtype='int',skip_header=1)


# In[3]:


train_data


# In[4]:


train_data.shape


# In[5]:


y_train_orig= train_data[:,0:1]


# In[6]:


y_train_orig


# In[7]:


y_train_orig.shape


# In[8]:


X_train_orig=np.delete(train_data,0,axis=1)


# In[9]:


X_train_orig.shape


# In[10]:


X_train=np.reshape(X_train_orig,(X_train_orig.shape[0],28,28,1))


# In[11]:


X_train.shape


# In[12]:


test_data


# In[13]:


test_data.shape


# In[14]:


X_test=np.reshape(test_data,(test_data.shape[0],28,28,1))


# In[15]:


X_test.shape


# In[16]:


X_train[6]


# In[17]:


from keras.utils.np_utils import to_categorical   

Y_train = to_categorical(y_train_orig, num_classes=10)


# In[18]:


X_train = X_train/255.
X_test = X_test/255.
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))


# In[19]:


model=Sequential()
model.add(Conv2D(32, kernel_size=(9,9), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))


# In[20]:


model.summary()


# In[21]:


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10,activation='sigmoid'))


# In[23]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, Y_train, batch_size=128, epochs=10)
y_predict=model.predict(X_test)
df_submit=pd.DataFrame({'ImageId': range(1, 28001),'Label':np.argmax(y_predict,axis=-1)})
df_submit.to_csv('Submission.csv',index=False)


# In[ ]:





# In[ ]:




