import datetime

import tensorflow as tf

from generate_model.import_images import import_images
from generate_model.properties import MODEL_PATH


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
        # flatten before feeding into Dense neural network.
        tf.keras.layers.Flatten(),
        # 512 neurons in the hidden layer
        tf.keras.layers.Dense(512, activation="relu"),
        # 15 = 15 different categories
        # softmas takes a set of values and effectively picks the biggest one. for example if the output layer has
        # [0.1,0.1,0.5,0.2,0.1], it will take it and turn it into [0,0,1,0,0]
        tf.keras.layers.Dense(35, activation="softmax")
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
                        epochs=50,
                        validation_data=val,
                        workers=10,
                        callbacks=tensorboard_callback,
                        validation_steps=20,  # 20 x 32 (batch size) = 640 images
                        )

# Save
tf.keras.models.save_model(
    model,
    MODEL_PATH,
    overwrite=True,
    include_optimizer=True,
    save_format="tf",
    signatures=None
)
model.save(r"model/fruit.h5")
