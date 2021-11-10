import os
import cv2
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk("C:/Users/User/Desktop/STUDIA/s5/POiWK/fruits-recognition/dataset/fruits_360_dataset/fruits/Test/"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img



train_path = "C:/Users/User/Desktop/STUDIA/s5/POiWK/fruits-recognition/dataset/fruits_360_dataset/fruits/Training/"
test_path = "C:/Users/User/Desktop/STUDIA/s5/POiWK/fruits-recognition/dataset/fruits_360_dataset/fruits/Test/"



image = train_path + "Limes/0_100.jpg"
img = cv2.imread(image)
cv2.imshow('image', img)
cv2.waitKey(0)

img = img_to_array(img)
img.shape()
