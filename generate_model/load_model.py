import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing import image

from generate_model.properties import MODEL_H5_PATH


def show_training_statistics(model):
    acc = model.history['acc']  # training accuracy scores from the model that has been trained
    val_acc = model.history['val_acc']  # validation accuracy scores from the model that has been trained
    loss = model.history['loss']  # training loss scores from the model that has been trained
    val_loss = model.history['val_loss']  # validation loss scores from the model that has been trained
    epochs = range(len(acc))  # x axis
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Scores')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'ro', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss Scores')
    plt.legend()
    plt.show()


def load_model():
    return tf.keras.models.load_model(
        MODEL_H5_PATH,
        custom_objects=None,
        compile=True
    )


def fruit_prediction(image_dir):
    img_list = os.listdir(image_dir)
    for fruits in img_list:
        predict(fruits, image_dir)


def predict(fruits, image_dir):
    path = os.path.join(image_dir, fruits)
    img = image.load_img(path, target_size=(150, 150))
    array = image.img_to_array(img)
    x = np.expand_dims(array, axis=0)
    vimage = np.vstack([x])
    img_classification = model.predict(vimage)
    print(img_classification, fruits)


model = load_model()
model.summary()
