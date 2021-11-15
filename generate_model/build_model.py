import tensorflow as tf

from generate_model.import_images import import_images
from generate_model.properties import MODEL_PATH


class CallbackOnEpochEpochEnd(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch == 1:
            print("\nCancelling training, epoch 1")
            self.model.stop_training = True


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


train, test = import_images()

model = prepare_model_to_learn()
model.summary()
model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])

# Train
fruit_model = model.fit(train, epochs=100, validation_data=test, verbose=1, callbacks=[CallbackOnEpochEpochEnd()], workers=10)

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