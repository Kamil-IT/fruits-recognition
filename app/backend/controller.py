import numpy as np
import tensorflow as tf
from flask import Flask, request, Response
from keras_preprocessing.image import ImageDataGenerator

app = Flask(__name__)

classes = {'Apple Braeburn': 0, 'Apple Crimson Snow': 1, 'Apple Golden 1': 2, 'Apple Golden 2': 3, 'Apple Golden 3': 4,
           'Apple Granny Smith': 5, 'Apple Pink Lady': 6, 'Apple Red 1': 7, 'Apple Red 2': 8, 'Apple Red 3': 9,
           'Apple Red Delicious': 10, 'Apple Red Yellow 1': 11, 'Apple Red Yellow 2': 12, 'Apricot': 13, 'Avocado': 14,
           'Avocado ripe': 15, 'Banana': 16, 'Banana Lady Finger': 17, 'Banana Red': 18, 'Beetroot': 19,
           'Blueberry': 20, 'Cactus fruit': 21, 'Cantaloupe 1': 22, 'Cantaloupe 2': 23, 'Carambula': 24,
           'Cauliflower': 25, 'Cherry 1': 26, 'Cherry 2': 27, 'Cherry Rainier': 28, 'Cherry Wax Black': 29,
           'Cherry Wax Red': 30, 'Cherry Wax Yellow': 31, 'Chestnut': 32, 'Clementine': 33, 'Cocos': 34, 'Corn': 35,
           'Corn Husk': 36, 'Cucumber Ripe': 37, 'Cucumber Ripe 2': 38, 'Dates': 39, 'Eggplant': 40, 'Fig': 41,
           'Ginger Root': 42, 'Granadilla': 43, 'Grape Blue': 44, 'Grape Pink': 45, 'Grape White': 46,
           'Grape White 2': 47, 'Grape White 3': 48, 'Grape White 4': 49, 'Grapefruit Pink': 50,
           'Grapefruit White': 51, 'Guava': 52, 'Hazelnut': 53, 'Huckleberry': 54, 'Kaki': 55, 'Kiwi': 56,
           'Kohlrabi': 57, 'Kumquats': 58, 'Lemon': 59, 'Lemon Meyer': 60, 'Limes': 61, 'Lychee': 62, 'Mandarine': 63,
           'Mango': 64, 'Mango Red': 65, 'Mangostan': 66, 'Maracuja': 67, 'Melon Piel de Sapo': 68, 'Mulberry': 69,
           'Nectarine': 70, 'Nectarine Flat': 71, 'Nut Forest': 72, 'Nut Pecan': 73, 'Onion Red': 74,
           'Onion Red Peeled': 75, 'Onion White': 76, 'Orange': 77, 'Papaya': 78, 'Passion Fruit': 79, 'Peach': 80,
           'Peach 2': 81, 'Peach Flat': 82, 'Pear': 83, 'Pear 2': 84, 'Pear Abate': 85, 'Pear Forelle': 86,
           'Pear Kaiser': 87, 'Pear Monster': 88, 'Pear Red': 89, 'Pear Stone': 90, 'Pear Williams': 91, 'Pepino': 92,
           'Pepper Green': 93, 'Pepper Orange': 94, 'Pepper Red': 95, 'Pepper Yellow': 96, 'Physalis': 97,
           'Physalis with Husk': 98, 'Pineapple': 99, 'Pineapple Mini': 100, 'Pitahaya Red': 101, 'Plum': 102,
           'Plum 2': 103, 'Plum 3': 104, 'Pomegranate': 105, 'Pomelo Sweetie': 106, 'Potato Red': 107,
           'Potato Red Washed': 108, 'Potato Sweet': 109, 'Potato White': 110, 'Quince': 111, 'Rambutan': 112,
           'Raspberry': 113, 'Redcurrant': 114, 'Salak': 115, 'Strawberry': 116, 'Strawberry Wedge': 117,
           'Tamarillo': 118, 'Tangelo': 119, 'Tomato 1': 120, 'Tomato 2': 121, 'Tomato 3': 122, 'Tomato 4': 123,
           'Tomato Cherry Red': 124, 'Tomato Heart': 125, 'Tomato Maroon': 126, 'Tomato Yellow': 127,
           'Tomato not Ripened': 128, 'Walnut': 129, 'Watermelon': 130}
classes = {value: key for (key, value) in classes.items()}
model = tf.keras.models.load_model(
    r"../../generate_model/model_131/fruit.h5",
    custom_objects=None,
    compile=True
)


def predict():
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test = test_gen.flow_from_directory('img/',
                                        target_size=(100, 100),
                                        shuffle=True,
                                        batch_size=34,
                                        color_mode='rgb',
                                        class_mode="categorical")  # multi-class classification.

    prediction = model.predict(test)

    return prediction


def get_top_3_elements(predictions):
    top_ids = []
    for i in range(3):
        id_top = np.argmax(predictions)
        if predictions[id_top] is not 0.0:
            top_ids.append(id_top)
            predictions[id_top] = 0
    return top_ids


@app.route('/recognize/image', methods=['POST', 'GET'])
def show_user():
    request.files['myfile'].save('img/test/image_to_predict.jpg')

    prediction = predict()
    elements = get_top_3_elements(prediction[0].copy())
    response = [f'{round(prediction[0][i] * 100, 1)}% - {classes[i]} \n' for i in elements]
    return Response(', '.join(response) + ' Main Page: file:///C:/studia/fruits-recognition/app/frontend/index.html')


app.run()
