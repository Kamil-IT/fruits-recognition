import numpy as np
import tensorflow as tf
from flask import Flask, request
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = tf.keras.models.load_model(
    r"../../generate_model/model/fruit.h5",
    custom_objects=None,
    compile=True
)


def predict():
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    img = image.load_img('image_to_predict.jpg', target_size=(100, 100), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    prediction = model.predict(img_preprocessed)

    return prediction


@app.route('/recognize/image', methods=['POST', 'GET'])
def show_user():
    request.files['myfile'].save('image_to_predict.jpg')

    prediction = predict()
    return str(prediction.tolist())


app.run()

print('sdasdada')

