import tensorflow as tf
import numpy as np
import random
from . import image_utils


def get_autoencoder():

    autoencoder = tf.keras.Sequential()
    autoencoder.add(tf.keras.layers.Conv2D(8, (3, 3), input_shape=(128, 128, 3), padding="same", activation="relu"))
    autoencoder.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    autoencoder.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    autoencoder.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(tf.keras.layers.UpSampling2D((2,2)))
    autoencoder.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(tf.keras.layers.UpSampling2D((2,2)))
    autoencoder.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
    autoencoder.add(tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.summary()

    return autoencoder


def get_encoder():
    autoencoder = tf.keras.Sequential()
    autoencoder.add(tf.keras.layers.Conv2D(8, (3, 3), input_shape=(128, 128, 3), padding="same", activation="relu"))
    autoencoder.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    autoencoder.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    autoencoder.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    
    autoencoder.compile(optimizer='adam', loss='mse')

    #autoencoder.summary()
    return autoencoder


def get_decoder():
    autoencoder = tf.keras.Sequential()
    autoencoder.add(tf.keras.layers.Conv2D(16, (3, 3), input_shape=(32, 32, 16), activation='relu', padding='same'))
    autoencoder.add(tf.keras.layers.UpSampling2D((2,2)))
    autoencoder.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(tf.keras.layers.UpSampling2D((2,2)))
    autoencoder.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
    autoencoder.add(tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    

    
    autoencoder.compile(optimizer='adam', loss='mse')

    #autoencoder.summary()
    return autoencoder

#Training
def fit_network(net, n_epochs=40):
    train_set_size = 800
    casual_array = np.arange(train_set_size).tolist()
     
    for i in range (0, n_epochs):
        train_set = []
        #40 images patch
        for j in range (0,40):
            index = random.randint(0, len(casual_array)-1)
            path_immagine = "dataset/train/" + str(casual_array[index]) + ".png" 
            image_rgb = image_utils.load_image(path_immagine)[0]
            pad_img = image_utils.pad_train_image(image_rgb)
            blocks = image_utils.get_train_blocks(pad_img)        
            train_set.extend(blocks)
            del(blocks)
            casual_array.pop(index)
            
        train_set = np.asarray(train_set).astype(float)
        net.fit(x=train_set, y=train_set, batch_size=128, epochs=1, verbose=1, shuffle=True, initial_epoch=0)
        del(train_set)
        print("Epoca " + str(i+1) + " su " + str(n_epochs) + 
              ". Mancano " + str(len(casual_array)) + " immagini")
        print() 
    return net



def get_trained_encoder_decoder(trained_autoencoder, n_layers=6):
    encoder = get_encoder()
    decoder = get_decoder()
    
    encoder.set_weights(trained_autoencoder.get_weights()[0:n_layers] )
    decoder.set_weights(trained_autoencoder.get_weights()[n_layers:] )
    
    return encoder, decoder
    
    
   
