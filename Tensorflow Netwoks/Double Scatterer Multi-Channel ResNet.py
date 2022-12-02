import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_addons.optimizers import CyclicalLearningRate
from tensorflow.keras.callbacks import Callback
from tensorflow import keras
from keras import layers
import FieldGen as fg
from FieldGen import *


class LRFinder(Callback):
    """`Callback` that exponentially adjusts the learning rate after each training batch between `start_lr` and
    `end_lr` for a maximum number of batches: `max_step`. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the `plot` method.
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 100, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)

def DataGen(r,n,batch_size):
    """This generates two arrays; one contains field values, generated using the imported file FieldGen, at points 
       along the circumference of a give circle. The other contains the corresponding positions of the scatterers for 
       that field.
    
       Inputs:
    
       r(int): Radius of circle
    
       n(int): Number of points
       
       batch_size(int): Number of unique arrays
       
       Outputs:
       
       Train_in(np.array): An array containing batches of normalized real and imaginary field values.
       
       Train_out(np.array): An array containg batches of cooridinates of scatterers.
    """
    
    train_in = np.zeros((batch_size,n,2),dtype=float)
    train_out = np.zeros((batch_size,2))
    
    for i in range(batch_size):
    
        P = np.random.uniform(2, r - 2, (1,2)) / (r-2) #Generates the coordinate of the scatterer, confined to a region within 
                                                       #the radius r. The values are normalized to aid the network training.
        train_out[i] = P[0]
        
        temp = fg.Circle_Field_Generator(r,n,P,1,2)[1]
        
        train_in[i,:,0] = temp.real / np.amax(temp.real) #The real and imaginary values are assigned to two seperate channels
        train_in[i,:,1] = temp.imag / np.amax(temp.imag) #and normalized by their maximum values.
        
        np.nan_to_num(train_in, copy=False, nan=0.0) #any nan values are removed as these woul affect the network training.
        
    return train_in, train_out

train_in, train_out = DataGen(20,500,10000)
test_in, test_out = DataGen(20,500,2000)
    

#Everything above is training data generation
train_in = train_in.reshape((10000,1,500))
test_in = test_in.reshape((2000,1,500))

shape_in = train_in[0].shape
shape_out = train_out[0].shape

#Defining network architecture

input_shape = train_in[0].shape

def get_resnet_model(categories=4):
    def residual_block(X, kernels, stride):
        out = keras.layers.Conv1D(kernels, stride, padding='same')(X)
        out = keras.layers.ReLU()(out)
        out = keras.layers.Conv1D(kernels, stride, padding='same')(out)
        out = keras.layers.add([X, out])
        out = keras.layers.ReLU()(out)
        out = keras.layers.MaxPool1D(2, 1)(out)
        return out

    kernels = 512
    stride = 3
    
    inputs = keras.layers.Input(input_shape)
    X = keras.layers.Conv1D(kernels, stride)(inputs)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(32, activation='relu')(X)
    X = keras.layers.Dense(32, activation='relu')(X)
    output = keras.layers.Dense(4, activation='relu')(X)

    model = keras.Model(inputs=inputs, outputs=output)
    return model

model = get_resnet_model(categories=4)

optimizer = keras.optimizers.Adam(lr=0.001)
model = get_resnet_model() 
model.compile(optimizer=optimizer, loss=ls.absolute_loss, metrics=[ls.absolute_loss])

lr_finder = LRFinder(start_lr=1e-7, end_lr= 1e-03, max_steps=100, smoothing=0.6)
_ = model.fit(train_in, train_out, batch_size=256, epochs=5, callbacks=[lr_finder], verbose=False)
lr_finder.plot()

# Set cyclical learning rate
N = train_in.shape[0]
batch_size = 1000
iterations = N/batch_size
step_size= 2 * iterations

lr_schedule = CyclicalLearningRate(1e-6, 1e-3, step_size=step_size, scale_fn=lambda x: tf.pow(0.95,x))
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

save_best_weights = ModelCheckpoint(filepath="weights.hdf5", verbose=0, save_best_only=True)

resnet_model = get_resnet_model() 
resnet_model.compile(optimizer=optimizer, loss=ls.absolute_loss, metrics=[ls.absolute_loss])
history = resnet_model.fit(train_in, train_out, validation_data=(test_in, test_out), 
                           shuffle=True, batch_size=batch_size, epochs=75, callbacks=[save_best_weights])

result = model.predict(test_in)

def swapper(test_out,result):
    """Function that ensures the results created by the network are orientated the same way as the true results"""
    
    a = result[0:2]
    
    d1 = np.linalg.norm(test_out[0:2] - a)
    d2 = np.linalg.norm(test_out[2:4] - a)
    
    if np.minimum(d1,d2) == d1:
        
        return result

    else:
        
        result[0:2] = result[2:4]
        result[2:4] = a
        
        return result
    
for i in range(200):
    
    result[i] = swapper(test_out[i],result[i])


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
