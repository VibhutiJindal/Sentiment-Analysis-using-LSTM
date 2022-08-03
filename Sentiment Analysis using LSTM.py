#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy==1.16.1')
import warnings
warnings.filterwarnings('ignore')


# In[16]:


#importing necessary libraries
import numpy as np
from numpy import array
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
import re
from nltk.tokenize import word_tokenize
import nltk


# In[3]:


#fix random seed for reproducability
np.random.seed(7)


# In[4]:


#loading the dataset but here we only keep the top n words and zero out the rest i.e keep vocabulary size as 5000
top_words = 5000 #vocabulary_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# In[5]:


print('---review---')
print(X_train[6])
print('---label---')
print(y_train[6])


# In[6]:


# using the dictionary returned by imdb.get_word_index() to map the review back to the original words
word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('review with words')
print([id2word.get(i, ' ') for i in X_train[6]])
print('label')
print(y_train[6])


# In[7]:


print(word2id)


# In[8]:


print(id2word)


# In[9]:


#Maximum review length and minimum review length.
print('Maximum review length: {}'.format(
len(max((X_train + X_test), key=len))))

print('Minimum review length: {}'.format(
len(min((X_train + X_test), key=len))))


# In[10]:


#In order to feed this data into our RNN, all input documents must have the same length. We will limit the maximum review length to maximum words by truncating longer reviews and padding shorter reviews with a null value (0). We can accomplish this task using the pad_sequences() function in Keras. Here, setting max_review_length to 500.'''
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# In[11]:


#our input is a sequence of words (integer word IDs) of maximum length = max_review_length, and our output is a binary sentiment label (0 or 1).


# In[12]:


# creating the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


# In[13]:


# compile our model by specifying the loss function and optimizer 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[14]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=20)


# In[15]:


#Calculate Accuracy
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:




