import cv2
import numpy as np
import time
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from keras.models import load_model

model = load_model('NN.h5')
classes = ['Healthy Lungs','Viral Pneumonia detected','Bacterial Pneumonia detected']

def get(file_path):

        img = cv2.imread(file_path)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(grayImage)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        height, width = img.shape[:2]
        img = cv2.resize(img, (100,100))

        # # predict!
        roi_X = np.expand_dims(img, axis=0)
        predictions = model.predict(roi_X)

        print(np.argmax(predictions[0]))
        result_index = np.argmax(predictions[0])
        result = classes[result_index]

        # # Get the output of the last convolutional layer
        # last_conv_layer = model.get_layer('dense_2')
        # # Get the gradients of the predicted class with respect to the last convolutional layer
        # grads = keras.backend.gradients(predictions[:, result_index], last_conv_layer.output)[0]

        # # Get the mean intensity values of the gradients along each channel axis
        # pooled_grads = keras.backend.mean(grads, axis=(0, 1, 2))

        # # Define a function to generate the heat map
        # iterate = keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        # conv_layer_output_value = last_conv_layer.output[0]
        # pooled_grads_value, conv_layer_output_value = iterate([img])
        # for i in range(pooled_grads.shape[0]):
        #         conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        
        # # Create the heat map
        # heatmap = np.mean(conv_layer_output_value, axis=-1)

        # # Normalize the heat map values between 0 and 1
        # heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        # # Resize the heat map to match the input image size
        # heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))

        # # Convert the heat map to RGB format
        # heatmap = np.uint8(255 * heatmap)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # # Overlay the heat map on the input image
        # superimposed_img = cv2.addWeighted(cv2.cvtColor(np.uint8(255 * img[0]), cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)
        # print(superimposed_img)




        cv2.imwrite('./static/images/p1.png',equalized)
        cv2.imwrite('./static/images/p2.png',hsv)
        # cv2.imwrite('./static/images/gradimg.png', heatmap)
        return result
        
        