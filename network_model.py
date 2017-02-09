from keras.layers import Dense, ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import Adam

image_size = 64

def NeuralNetwork(input_shape):
    
    model = Sequential()

    # 1st CNN Layer with Maxpooling and dropout
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    #model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
    
    # 2nd CNN Layer with Maxpooling and dropout
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 3rd CNN Layer with Maxpooling and dropout
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
    
    # 4th CNN Layer with Maxpooling and dropout. Strides are changed to (1,1) from this layer onwards
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))    

    # 5th CNN Layer with Maxpooling and dropout
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # 6th CNN Layer with Maxpooling and dropout
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))    

    # Below are few Fully connected layers
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1164, activation='relu'))
    # Dropout here to avoid overfitting
    # model.add(Dropout(0.5))    
    model.add(Dense(256, activation='relu'))
    # Last Dropout to avoid overfitting
    #model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))    
    model.add(Dense(1))

    return model