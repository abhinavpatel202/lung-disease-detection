import cv2
import numpy as np
import time
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

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

        cv2.imwrite('./static/images/p1.png',equalized)
        cv2.imwrite('./static/images/p2.png',hsv)
        return result
        
