import os
from glob import glob

import cv2
import numpy as np

from keras import preprocessing
##from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical


import matplotlib.pyplot as plt

# get the reference to the webcam
width  = 100
height = 100

##############

def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.png')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)

        images.append(x)
    return images

###############

print("Normal")
a = load_images('./data/NORMAL')
print("virus")
b = load_images('./data/virus')
print('bacteria')
c = load_images('./data/bacteria')
print("Done")

###################

# convert into numpy array
a = np.array(a)
b = np.array(b)
c = np.array(c)

X = np.concatenate((a,b,c), axis=0)

##############

# normalization
X = X / 255.

###################


ya = [0 for item in enumerate(a)]
yb = [1 for item in enumerate(b)]
yc = [2 for item in enumerate(c)]

y = np.concatenate((ya,yb,yc), axis=0)

y = to_categorical(y, num_classes=3)

#print(y.shape)


#####################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
##from keras.optimizers import Adam

from tensorflow.keras.optimizers import Adam



# default parameters
conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2
lr = 0.001

epochs = 500
batch_size = 32
color_channels = 3

def build_model(conv_1_drop=conv_1_drop, conv_2_drop=conv_2_drop,
                dense_1_n=dense_1_n, dense_1_drop=dense_1_drop,
                dense_2_n=dense_2_n, dense_2_drop=dense_2_drop,
                lr=lr):
    model = Sequential()

    model.add(Convolution2D(conv_1, (5, 5),
                            input_shape=(width, height, color_channels),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_1_drop))

    model.add(Convolution2D(conv_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_2_drop))
        
    model.add(Flatten())
        
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))

    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr),
                  metrics=['accuracy'])

    return model

#######################


# model with base parameters
model = build_model()

model.summary()

#################
epochs = 100
##################

trained_model = model.fit(X, y, epochs=epochs)

model.save('NN.h5')



plt.plot(trained_model.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('model_accuracy.png')
plt.show()

plt.clf()

plt.plot(trained_model.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_loss.png')
plt.show()

print("Model_Ready")
