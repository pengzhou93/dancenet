
# coding: utf-8

# In[2]:


from model import vae,decoder
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Input
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import mdn
from sklearn.preprocessing import MinMaxScaler


# # Set paths

# In[3]:


ENCODED_DATA_PATH = 'models/data/lv.npy'
VAE_PATH = 'models/weights/vae_cnn.h5'
DANCENET_PATH = 'models/weights/gendance.h5'


# # Load encoded data

# In[4]:


data = np.load(ENCODED_DATA_PATH)
print(data.shape)


# # Normalize data

# In[5]:


data = np.array(data).reshape(-1,128)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(data)
data =  scaler.transform(data)


# In[6]:


numComponents = 24
outputDim = 128


# # LSTM + MDN 

# In[7]:


inputs = Input(shape=(128,))
x = Reshape((1,128))(inputs)
x = LSTM(512, return_sequences=True,input_shape=(1,128))(x)
x = Dropout(0.40)(x)
x = LSTM(512, return_sequences=True)(x)
x = Dropout(0.40)(x)
x = LSTM(512)(x)
x = Dropout(0.40)(x)
x = Dense(1000,activation='relu')(x)
outputs = mdn.MDN(outputDim, numComponents)(x)
model = Model(inputs=inputs,outputs=outputs)
print(model.summary())


# In[8]:


opt = adam(lr=0.0005)
model.compile(loss=mdn.get_mixture_loss_func(outputDim,numComponents),optimizer=opt)


# In[9]:


train = False #change to True to train from scratch

if train:
    X = data[0:len(data)-1]
    Y = data[1:len(data)]
    checkpoint = ModelCheckpoint(DANCENET_PATH, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    model.fit(X,Y,batch_size=1024, verbose=1, shuffle=False, validation_split=0.20, epochs=10000, callbacks=callbacks_list)


# # Load weights

# In[10]:


vae.load_weights(VAE_PATH)
model.load_weights(DANCENET_PATH)


# # Generate Video

# In[11]:


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("out.mp4", fourcc, 30.0, (208, 120))
lv_in = data[0]

for i in range(500):
    input = np.array(lv_in).reshape(1,128)
    lv_out = model.predict(input)
    shape = np.array(lv_out).shape[1]
    lv_out = np.array(lv_out).reshape(shape)
    lv_out = mdn.sample_from_output(lv_out,128,numComponents,temp=0.01)
    lv_out = scaler.inverse_transform(lv_out)
    img = decoder.predict(np.array(lv_out).reshape(1,128))
    img = np.array(img).reshape(120,208,1)
    img = img * 255
    img = np.array(img).astype("uint8")
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    lv_in = lv_out
    video.write(img)
video.release()

