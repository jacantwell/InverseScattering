import os
import tensorflow as tf
from tensorflow import keras
import FieldGen as fg
from FieldGen import *

def DataGen(i):
    
    n = 1000
    r = 25
    stepr = 5
    c = 1
    x = [1,2,3,4]
    temp = np.zeros((c,n))
    
    P = np.random.randint(0, 20, (2,2)) / 20
    x[0:2] = P[0]
    x[2:4] = P[1]
    
    for k in range(c):

        temp[k] = fg.Circle_Field_Generator(r,n,P,1,2)[1]
        temp[k] = temp[k].real / np.amax(temp[k].real)
        
        r += stepr
        
    return  temp, x

batch_size = 10000

def DataGen(i):
    
    n = 1000
    r = 25
    x = [1,2,3,4]
    
    P = np.random.randint(2, r - 2, (2,2))
    x[0:2] = P[0]
    x[2:4] = P[1]
        
    temp = fg.Circle_Field_Generator(r,n,P,2,0.1)[1]
    temp = temp.real / (2*np.amax(temp.real))
        
    return  temp, x

data = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
        delayed(DataGen)(num)
        for num in range(batch_size)
    )

data = np.array(data,dtype=object)
train_in = np.zeros((batch_size,1000))
train_out = np.zeros((batch_size,4))

for i in range(batch_size):

    train_in[i] = data[i,0]
    train_out[i] = data[i,1]
    
data = Parallel(n_jobs=-1, prefer="processes", verbose=6)(
        delayed(DataGen)(num)
        for num in range(int(batch_size/5))
    )

data = np.array(data,dtype=object)
test_in = np.zeros((int(batch_size/5),1000))
test_out = np.zeros((int(batch_size/5),4))

for i in range(int(batch_size/5)):

    test_in[i] = data[i,0]
    test_out[i] = data[i,1]
    

#Everything above is training data generation

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

model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate=0.00001),metrics=['mean_absolute_error'])
epochs = 50

history = model.fit(train_in, train_out, batch_size=10000, epochs=epochs, shuffle=False, verbose=1, validation_data=(test_in, test_out))


result = model.predict(test_in)

#Plotting network result vs actual result

x_data = test_out[:,0]
y_data = result[:,0]
x1_data = test_out[:,1]
y1_data = result[:,1]

plt.figure(figsize=(10,10))
plt.scatter(x_data,y_data, label='X Values')
plt.scatter(x1_data,y1_data,c='r',label='Y Values')
plt.xlabel('Actual Value')
plt.xticks(np.arange(0,18 / 20,1 / 20))
plt.ylabel("Network's Predicted Value")
plt.yticks(np.arange(0,18 / 20,1 / 20))
plt.legend()

#Plotting absolute error during training over epoch number

loss_train = history.history['mean_absolute_error']
loss_val = history.history['val_mean_absolute_error']
epochs = range(1,epochs+1)
plt.figure(figsize=(10,10))
plt.plot(epochs, loss_train, 'g', label='Training Absolute Error',linewidth=1)
plt.plot(epochs, loss_val, 'b', label='Validation Absolute Error',linewidth=1)
plt.title('Training and Validation Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()
