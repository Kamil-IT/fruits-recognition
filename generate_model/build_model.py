import datetime

import tensorflow as tf

# from generate_model.import_images import import_images
# from generate_model.properties import MODEL_PATH

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# from generate_model.properties import TRAIN_DIR, VAL_DIR, TEST_DIR

import os
cwd = os.getcwd()


TRAIN_DIR = os.path.join(cwd, 'dataset','without_background','fruits','Training')
VAL_DIR = os.path.join(cwd, 'dataset','without_background','fruits','Validate')
TEST_DIR = os.path.join(cwd, 'dataset','without_background','fruits','Test')

MODEL_PATH = r"model"
MODEL_H5_PATH = r"model/fruit.h5"


def import_images():
    training_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train = training_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(100, 100),
                                                 shuffle=True,
                                                 batch_size=34,
                                                 color_mode='rgb',
                                                 class_mode="categorical")  # multi-class classification.

    val = validation_datagen.flow_from_directory(VAL_DIR,
                                                 target_size=(100, 100),
                                                 shuffle=True,
                                                 color_mode='rgb',
                                                 class_mode="categorical")  # multi-class classification.

    test = test_datagen.flow_from_directory(TEST_DIR,
                                            target_size=(100, 100),
                                            color_mode='rgb',
                                            batch_size=49,
                                            class_mode="categorical",  # multi-class classification.
                                            shuffle=False)

    print(train.class_indices)
    print(val.class_indices)
    print(test.class_indices)
    return train, val, test


def prepare_model_to_learn():
    return tf.keras.models.Sequential([
        # first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # third convolution layer
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # fourth convolution layer
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # fifth convolution layer
        tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # flatten before feeding into Dense neural network.
        tf.keras.layers.Flatten(),
        # 512 neurons in the hidden layer
        tf.keras.layers.Dense(512, activation="relu"),
        # 15 = 15 different categories
        # softmas takes a set of values and effectively picks the biggest one. for example if the output layer has
        # [0.1,0.1,0.5,0.2,0.1], it will take it and turn it into [0,0,1,0,0]
        tf.keras.layers.Dense(131, activation="softmax")
    ])


train, val, test = import_images()

model = prepare_model_to_learn()
model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer='rmsprop',
    metrics=[
        'accuracy',
        'FalseNegatives',
        'FalsePositives',
        'TopKCategoricalAccuracy',
        'TrueNegatives',
        'TruePositives'
    ])


def get_log_folder_name(n_classes):
    return f'log/{n_classes}-classes_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'


# Metrics
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=get_log_folder_name(train.num_classes),
    histogram_freq=1)

# Train
fruit_model = model.fit(train,
                        epochs=5,
                        validation_data=val,
                        workers=10,
                        callbacks=tensorboard_callback,
                        validation_steps=20,  # 20 x 34 (batch size) = 640 images
                        )

# # Save
# tf.keras.models.save_model(
#     model,
#     MODEL_PATH,
#     overwrite=True,
#     include_optimizer=True,
#     save_format="tf",
#     signatures=None
# )
# model.save(r"model/fruit.h5")
