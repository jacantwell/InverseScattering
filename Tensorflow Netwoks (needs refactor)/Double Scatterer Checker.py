import os
import tensorflow as tf
from tensorflow import keras
import FieldGen as fg
from FieldGen import *
import LossFunctions as ls
import random
import joblib
from joblib import Parallel, delayed

def absolute_error(y_in,y_out):
    """Returns absolute error"""

    mae = tf.keras.losses.MeanAbsoluteError()
    d1 = tf.abs(mae([y_in[0],y_in[1]], [y_out[0],y_out[1]]))
    d2 = tf.abs(mae([y_in[0],y_in[1]], [y_out[2],y_out[3]]))

    return tf.math.minimum(d1,d2)

def DataGen(i):
    
    n = 800
    r = 25
    R = np.zeros((4))
    #temp = np.zeros((n))
    
    x = np.array(random.sample(range(0, 20), 2), dtype=np.int)
    P = x.reshape((1,2))
    print(P)
    R[:2] = x
    R[2:4] = x
    
    temp = fg.Circle_Field_Generator(r,n,P,1,2)[1].real
        
    return  temp, (R/20)

data = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
        delayed(DataGen)(num)
        for num in range(8000)
    )

data = np.array(data,dtype=object)
train_in = np.zeros((8000,800))
train_out = np.zeros((8000,4))

for i in range(8000):

    train_in[i] = data[i,0]
    train_out[i] = data[i,1]

data = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
        delayed(DataGen)(num)
        for num in range(1000)
    )

data = np.array(data,dtype=object)
test_in = np.zeros((1000,800))
test_out = np.zeros((1000,4))

for i in range(1000):

    test_in[i] = data[i,0]
    test_out[i] = data[i,1]

a = np.max(train_in)
b = np.max(test_in)

train_in = train_in / np.max([a,b])
test_in = test_in / np.max([a,b])


shape_in = train_in[0].shape
shape_out = train_out[0].shape

#Defining network architecture

model = keras.Sequential([
    
    keras.layers.Input(shape=shape_in), 
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
])

print(model.summary())



model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=['mean_absolute_error'])
epochs = 500

history = model.fit(train_in, train_out, batch_size=8000, epochs=epochs, shuffle=True, verbose=1, validation_data=(test_in, test_out))

result = model.predict(test_in)

x1data = test_out[:,0] 
x1pred = result[:,0]
y1data = test_out[:,1] 
y1pred = result[:,1]

x2data = test_out[:,2] 
x2pred = result[:,2]
y2data = test_out[:,3] 
y2pred = result[:,3]

plt.figure(figsize=(10,10))
plt.scatter(x1data,x1pred, label='First X Value')
plt.scatter(x2data,x2pred, label='Second X Value')

plt.scatter(y1data,y1pred, label='First Y Value')
plt.scatter(y2data,y2pred, label='Second Y Value')

plt.xlabel('Actual Value')
plt.xticks(np.arange(0,1,0.1))
plt.ylabel("Network's Predicted Value")
plt.yticks(np.arange(0,1,0.1))

plt.legend()

loss_train = history.history['absolute_error']
loss_val = history.history['val_absolute_error']
epochs = range(1,501)
plt.figure(figsize=(10,10))
plt.plot(epochs, loss_train, 'g', label='Training Absolute Error',linewidth=1)
plt.plot(epochs, loss_val, 'b', label='Validation Absolute Error',linewidth=1)
plt.title('Training and Validation Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()



