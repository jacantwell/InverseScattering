#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow import keras
import FieldGen as fg
from FieldGen import *

def DataGen(size,step,batch_size):
    
    n = 1
    length = int((size/step) + 1)
    
    train_in = np.zeros((batch_size,length,length),dtype=complex)
    train_out = np.zeros((batch_size,length,length))
    
    for i in range(0,batch_size):
    
        P = np.random.randint(0,size - 1,(1,2))
        indeX = int(P[0,0] / step)
        indeY = int(P[0,1] / step)

        train_out[i,indeY,indeX] = 1
        
        train_in[i] = fg.Field_Generator(size,step,P,np.random.randint(0,10),np.random.randint(0,10))[2]
        
        np.nan_to_num(train_in, copy=False, nan=0.0)
        
    return train_in, train_out

train_in,train_out = DataGen(10,0.1,100)
test_in, test_out = DataGen(10,0.1,50)

model = keras.Sequential([
    
    keras.layers.Input(shape=shape_in),        
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'), 
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(shape_in[0], activation='relu'),
])

model.summary()

loss=tf.keras.losses.BinaryCrossentropy()
optim = keras.optimizers.Adam(learning_rate=0.01)
metrics = ["accuracy"]   
model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 100
epochs = 10

model.fit(train_in, train_out, batch_size=batch_size, epochs=epochs, shuffle=False, verbose=2)

test_loss, test_acc = model.evaluate(test_in,  test_out, verbose=2)
print('\nTest accuracy:', test_acc)

pred_in, pred_out = DataGen(10,0.1,1)
result = model.predict(pred_in)

print(np.where(result[0] > 0))
print(np.where(pred_out[0] > 0))
