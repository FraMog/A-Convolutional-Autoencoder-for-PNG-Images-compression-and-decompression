import tensorflow as tf
import numpy as np

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


def fit_network(net, block_size, n_epochs=40):
    train_set_size=800
    casual_array=np.arange(train_set_size).tolist()
    
    
    for i in range (0, n_epochs):
        train_set=[]
        for j in range (0,40):
            index=randint(0, len(casual_array)-1)
            path_immagine= "dataset2_preloaded/train/" + str(casual_array[index]) + ".png" 
            blocks = image_utils.get_blocks(image_utils.load_image(path_immagine), block_size)
        
            train_set.extend(blocks)

            del(blocks)
            print(index)
            casual_array.pop(index)
            
        train_set = np.asarray(train_set).astype(float)
      
                                              
        nz_train_set = train_set
        del(train_set)
        net.fit(x=nz_train_set, y=nz_train_set, batch_size=128, epochs=1, verbose=1, shuffle=True, initial_epoch=0)
        del(nz_train_set)
        print("Mancano " + str(len(casual_array)) + " immagini")
        
    return net

    
#Predizione su un blocco di un'immagine nuova
def get_block_prediction(index, net, blocks):
    pred=net.predict(np.asarray([blocks[index]]))
    return pred

def get_trained_encoder_decoder(trained_autoencoder, n_layers=6):
    encoder = get_encoder()
    decoder = get_decoder()
    
    encoder.set_weights(trained_autoencoder.get_weights()[0:n_layers] )
    decoder.set_weights(trained_autoencoder.get_weights()[n_layers:] )
    
    return encoder, decoder
    