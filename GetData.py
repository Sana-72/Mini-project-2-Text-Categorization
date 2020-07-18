#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import os

import itertools
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
layers = keras.layers
models = keras.models


# This code was tested with TensorFlow v1.8
print("You have TensorFlow version", tf.__version__)


# In[3]:


data = pd.read_csv(r"C:\Users\DOCTOR\Desktop\bbc-text.csv")


# In[4]:


data.head()


# In[5]:


data['category'].value_counts()


# In[6]:


train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))


# In[7]:


def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test


# In[ ]:




