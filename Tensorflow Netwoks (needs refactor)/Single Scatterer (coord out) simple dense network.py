import os
import tensorflow as tf
from tensorflow import keras
import FieldGen as fg
from FieldGen import *

def DataGen(size,step,batch_size):

    n = 1
    length = int((size/step) + 1)

    train_in = np.zeros((batch_size,length,length,2),dtype=float)
    train_out = np.zeros((batch_size,2))

    for i in range(0,batch_size):

        P = np.random.randint(0,size - 1,(1,2)) / 10

        train_out[i] = P[0]

        temp = fg.Field_Generator(size,step,P,2,20)[2]

        train_in[i,:,:,0] = temp.real
        train_in[i,:,:,1] = temp.imag

        np.nan_to_num(train_in, copy=False, nan=0.0)

    return train_in, train_out

train_in, train_out = DataGen(10,0.5,1000)
test_in, test_out = DataGen(10,0.5,100)

shape_in = train_in[0].shape
shape_out = train_out[0].shape
model = keras.Sequential([

    keras.layers.Flatten(input_shape=shape_in),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),


    keras.layers.Dense(2, activation='sigmoid'),
])

print(model.summary())

#setting loss and optimizer
loss='poisson'
optim = keras.optimizers.Adam(learning_rate=0.0001)    #Adam is the type of optimizer and lr is the learning rate
metrics = ["mean_absolute_error"]    #defining metric we want to track
metric = metrics[0]
model.compile(loss=loss, optimizer=optim, metrics=metrics)

history = model.fit(train_in, train_out, batch_size=500, epochs=100, shuffle=False, verbose=2, validation_data=(test_in, test_out))

result = model.predict(test_in)
diff = np.zeros(len(result))
for i in range(len(result)):
    diff[i] = np.linalg.norm((test_out[i] - result[i]))

plt.scatter(np.arange(0,100,1),diff)
plt.ylabel('Error')
plt.xlabel('Results')

x_data = np.concatenate([test_out[:,0],test_out[:,1]])
y_data = np.concatenate([result[:,0],result[:,1]])

plt.scatter(x_data,y_data)

loss_train = history.history['mean_absolute_error']
loss_val = history.history['val_mean_absolute_error']
epochs = range(1,101)
plt.plot(epochs, loss_train, 'g', label='Training Absolute Error')
plt.plot(epochs, loss_val, 'b', label='Validation Absolute Error')
plt.title('Training and Validation Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()
