# Importing all the libraries
import pandas as pd
import numpy as np
import math
import cv2
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Dense, ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import Adam

# Loading the data and splitting them into trianing and validation set.
driving_data = pd.read_csv('../driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])
steering_data = driving_data['steering'].values

X_train, X_val, y_train, y_val = train_test_split(driving_data, steering_data, test_size=0.2, random_state=42)
print('train input size {}'.format(len(X_train)))
print('train input result {}'.format(len(y_train)))
print('val input size {}'.format(len(X_val)))
print('val input result {}'.format(len(y_val)))


# parameters for this model
image_size = 64     # as we will be making square images for easier processing
batch_size = 256
epochs = 5

# Helper Methods
# randomizing the brightness in the image. As pointed out on slack randomizing brightness help the network learn faster and more accurately
def randomize_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_brightness = .1 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

# pre-processing each image - random brightness, normalization and resizing
def process_image(image_path):
    image = cv2.imread(image_path.strip())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # randomizing the brightness images. It has been seen to improve the learning the network does on these images.
    image = randomize_brightness(image)
    shape = image.shape
    # normalizing and resizing the image
    image = image[math.floor(shape[0] / 5.):shape[0] - 25, 0:shape[1]]
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image / 255.

    return np.array(image)


# since data being loaded is too big adding a generator for the same
def data_generator(input_data):
    
    # shuffling the data
    input_data = shuffle(input_data)

    # creating holders for images and target steering values
    x_images = np.zeros((batch_size, image_size, image_size, 3))
    y_val = np.zeros(batch_size)

    while True:
        for i in range(batch_size):
            
            line_number = np.random.randint(len(input_data))            
            image_row = input_data.iloc[[line_number]].reset_index()

            # randomly picking up center, left or right image for the batch.
            # purpose of this is to increase the randomness in the training set.
            # for the left and right images, it being shifted a bit to componsate for its position.
            # 0.2 to 0.4 is what many have tried. I just chose a mid-value for my model.
            rand = np.random.randint(3)
            if (rand == 0):
                image_path = image_row['left'][0].strip()
                shift_angle = 0.3
            elif (rand == 1):
                image_path = image_row['right'][0].strip()
                shift_angle = -0.3
            else:
                image_path = image_row['center'][0].strip()
                shift_angle = 0.0

            # adding images to the input images
            x_images[i] = process_image(image_path)
            y_val[i] = image_row['steering'][0] + shift_angle

        yield x_images, y_val


# The network that is being created here is inspired by the Nvidia model.
def NeuralNetwork_nvidia(input_shape):
    
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




'''
main run methods

Here I am defining the following things:
1. shape of input
2. creating a network model object
3. setting up the optimizer as Adam and setting learning-rate as 0.0001
4. setting error function as MSE
5. using generator to create two separate generator for training and validation set
6. training the model
7. once the training completes, saving the model and weights in model.json and model.h5 files respectively
'''
input_shape = (3, image_size, image_size)

model = NeuralNetwork_nvidia(input_shape)
opt = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=opt)

train_generator = data_generator(X_train)
train_validator = data_generator(X_val)
model.fit_generator(train_generator, samples_per_epoch=len(y_train), nb_epoch=epochs, validation_data=train_validator, nb_val_samples=len(y_train) / 6)

    
# Saving the trained model into model.json file and related weights into model.h5
with open('model.json', 'w') as fd:
    json.dump(model.to_json(), fd)

model.save_weights('model.h5')
print('model and its weights are saved.')


