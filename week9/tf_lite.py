#!/usr/bin/env python
# coding: utf-8

# In[1]:

# run in terminal
# jupyter nbconvert --to script tf_lite.ipynb

get_ipython().run_line_magic('autosave', '0')


# In[1]:


get_ipython().system('python -V')


# In[3]:


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# In[4]:


import tensorflow as tf
from tensorflow import keras
tf.__version__


# In[7]:


model = keras.models.load_model('xception_v4_1_06_0.894.h5') 


# In[8]:


import numpy as np
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.applications.xception import preprocess_input


# In[10]:


load_img('pants.jpg', target_size=(150, 150))


# In[11]:


img = load_img('pants.jpg', target_size=(299, 299))

x = np.array(img)
X = np.array([x])

X = preprocess_input(X)


# In[12]:


X.shape


# In[14]:


preds = model.predict(X)
preds


# In[15]:


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]
dict(zip(classes, preds[0]))


# #### TF-Lite 
# 
# __Convert Keras into TF-Lite__

# In[17]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('clothing-model-1.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# In[18]:


get_ipython().system('ls -lh')


# ### Use TF-Lite

# In[20]:


import tensorflow.lite as tflite


# In[21]:


interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
# memory allocation
interpreter.allocate_tensors()


# In[22]:


interpreter.get_input_details()


# In[23]:


interpreter.get_input_details()[0] # pull index -> 0


# In[24]:


input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[25]:


# make predictions
interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[26]:


# compare with previous results
classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))


# ### Remove all TF dependencies 

# In[31]:


from PIL import Image


# In[35]:


with Image.open('pants.jpg') as img:
    img = img.resize((299, 299), Image.Resampling.NEAREST)


# In[36]:


img


# Find the way to replace
# 
# ```python
# from tensorflow.keras.preprocessing.image import load_img 
# from tensorflow.keras.applications.xception import preprocess_input
# ```

# In[37]:


x = np.array(img, dtype='float32')
X = np.array([x])

X = preprocess_input(X)


# In[38]:


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


# In[40]:


X = preprocess_input(X)


# In[41]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[42]:


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))


# #### Using `keras-image-helper`

# In[43]:


get_ipython().system('pip install keras-image-helper')


# In[44]:


from keras_image_helper import create_preprocessor


# In[45]:


preprocessor = create_preprocessor("xception", target_size=(299, 299))


# In[46]:


url = 'http://bit.ly/mlbookcamp-pants'
X = preprocessor.from_url(url)


# ### Import TF-Lite

# In[48]:


get_ipython().system('pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime')


# In[1]:


import tflite_runtime.interpreter as tflite


# In[2]:


from keras_image_helper import create_preprocessor


# In[3]:


# model from laptop
interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[4]:


# model from saturn cloud
interpreter1 = tflite.Interpreter(model_path='clothing-model-1.tflite')
interpreter1.allocate_tensors()

input_index1 = interpreter1.get_input_details()[0]['index']
output_index1 = interpreter1.get_output_details()[0]['index']


# In[5]:


preprocessor = create_preprocessor('xception', target_size=(299, 299))
url = 'http://bit.ly/mlbookcamp-pants'
X = preprocessor.from_url(url)


# In[6]:


# laptop
interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[7]:


# saturn cloud
interpreter1.set_tensor(input_index1, X)
interpreter1.invoke()
preds1 = interpreter1.get_tensor(output_index1)


# In[8]:


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))


# In[9]:


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds1[0]))

