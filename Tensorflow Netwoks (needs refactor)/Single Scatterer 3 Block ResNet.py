import os
import tensorflow as tf
from tensorflow import keras
import FieldGen as fg
from FieldGen import *

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

def build_model(input_shape, nb_classes):
        n_feature_maps = 128

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        #gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='relu')(output_block_3)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        return model
    
model = build_model(train_in[0].shape,2)

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
