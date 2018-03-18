
# coding: utf-8

# In[3]:


import numpy as np
list_sentences_train=np.load('123.npy')


# In[7]:


list_sentences_test=np.array(["wowwowowow"])


# In[8]:


list_sentences_test.size


# In[13]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen=100
max_features = 15352 # how many unique words to use (i.e num rows in embedding vector)



tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[14]:


from keras.models import load_model

model = load_model('my_model.h5')


y_test = model.predict([X_te], batch_size=1024, verbose=1)


y_classes = y_test.argmax(axis=-1)



# In[16]:


y_classes[0]

